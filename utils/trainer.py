# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import overall_accuracy, fast_hist, per_class_iu, per_class_accuracy

import cupy as cp

import ctypes
from ctypes import cdll
from numpy.ctypeslib import ndpointer

class TrainingManager:
    def __init__(
        self,
        net,
        loss,
        loader_train,
        loader_val,
        train_sampler,  # If provided, we assume distributed training
        optim,
        scheduler,
        max_epoch,
        path,
        rank,
        world_size,
        fp16=True,
        class_names=None,
        tensorboard=True,
    ):
        # Optim. methods
        self.optim = optim
        self.fp16 = fp16
        self.scaler = GradScaler() if fp16 else None
        self.scheduler = scheduler

        # Dataloaders
        self.max_epoch = max_epoch
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.train_sampler = train_sampler
        self.class_names = class_names

        # Network
        self.net = net
        self.rank = rank
        self.world_size = world_size
        print(f"Trainer on gpu: {self.rank}. World size:{self.world_size}.")

        # Loss
        self.loss = loss

        # Checkpoints
        self.best_miou = 0
        self.current_epoch = 0
        self.path_to_ckpt = path

        ## KDtree

        lib = cdll.LoadLibrary('cudastuff/build/libmylib_train.so')
        self.training_kdtree_obj = lib.MyFastKDTree_new()
        self.runTree_training = lib.MyFastKDTree_run

        self.runTree_training.argtypes = [ctypes.c_void_p,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## indatav
                ctypes.c_size_t,                                  ## tree_size  
                ctypes.POINTER(ctypes.c_int)]                     ## d_results_k

        self.runTree_training.restype   = ctypes.c_void_p

        

        lib = cdll.LoadLibrary('cudastuff/build/libmylib_user.so')
        self.valid_kdtree_obj = lib.MyFastKDTree_new()

        self.runTree_valid = lib.MyFastKDTree_run

        self.runTree_valid.argtypes = [ctypes.c_void_p,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## indatav
                ctypes.c_size_t,                                  ## tree_size  
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## queries_k
                ctypes.c_size_t,                                  ## numQueries_k
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  ## queries_1
                ctypes.c_size_t,                                  ## numQueries_1
                ctypes.POINTER(ctypes.c_int),                     ## d_results_k
                ctypes.POINTER(ctypes.c_int),]                    ## d_results_1

        self.runTree_valid.restype   = ctypes.c_void_p

        # Monitoring
        if tensorboard and (self.rank == 0 or self.rank is None):
            self.writer_train = SummaryWriter(
                path + "/tensorboard/train/",
                purge_step=self.current_epoch * len(self.loader_train),
                flush_secs=30,
            )
            self.writer_val = SummaryWriter(
                path + "/tensorboard/val/",
                purge_step=self.current_epoch,
                flush_secs=30,
            )
        else:
            self.writer_val = None
            self.writer_train = None

    def print_log(self, running_loss, oAcc, mAcc, mIoU, ious):
        if self.rank == 0 or self.rank is None:
            # Global score
            log = (
                f"\nEpoch: {self.current_epoch:d} :\n"
                + f" Loss = {running_loss:.3f}"
                + f" - oAcc = {oAcc:.1f}"
                + f" - mAcc = {mAcc:.1f}"
                + f" - mIoU = {mIoU:.1f}"
            )
            print(log)
            # Per class score
            log = ""
            for i, s in enumerate(ious):
                if self.class_names is None:
                    log += f"Class {i}: {100 * s:.1f} - "
                else:
                    log += f"{self.class_names[i]}: {100 * s:.1f} - "
            print(log[:-3])
            # Recall best mIoU
            print(f"Best mIoU was {self.best_miou:.1f}.")

    def gather_scores(self, list_tensors):
        if self.rank == 0:
            tensor_reduced = [
                [torch.empty_like(t) for _ in range(self.world_size)]
                for t in list_tensors
            ]
            for t, t_reduced in zip(list_tensors, tensor_reduced):
                torch.distributed.gather(t, t_reduced)
            tensor_reduced = [sum(t).cpu() for t in tensor_reduced]
            return tensor_reduced
        else:
            for t in list_tensors:
                torch.distributed.gather(t)

    def one_epoch(self, training=True):
        # Train or eval mode
        if training:
            net = self.net.train()
            loader = self.loader_train
            if self.rank == 0 or self.rank is None:
                print("\nTraining: %d/%d epochs" % (self.current_epoch, self.max_epoch))
            writer = self.writer_train
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.current_epoch)
        else:
            net = self.net.eval()
            loader = self.loader_val
            if self.rank == 0 or self.rank is None:
                print(
                    "\nValidation: %d/%d epochs" % (self.current_epoch, self.max_epoch)
                )
            writer = self.writer_val
        print_freq = np.max((len(loader) // 10, 1))

        # Stat.
        running_loss = 0.0
        confusion_matrix = 0

        # Loop over mini-batches
        if self.rank == 0 or self.rank is None:
            bar_format = "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}"
            loader = tqdm(loader, bar_format=bar_format)
        for it, batch in enumerate(loader):

            
            Nmax = np.max([f.shape[0] for f in batch["pc"]])
            neigs = []
            upsample = []

            if training:
                for pc, pc_orig in zip(batch["pc"], batch["pc_orig"]):
                    N = pc.shape[0]
                    data = pc[:, :3].reshape(-1).astype(np.float32)
                    results_k = cp.zeros((N, 17), dtype=cp.int32)
                    results_k_ctypes = ctypes.cast(results_k.data.ptr, ctypes.POINTER(ctypes.c_int32))

                    self.runTree_training(self.training_kdtree_obj, data, data.size, results_k_ctypes)

                    results_k = torch.from_dlpack(results_k.toDlpack()).long()
                    new_arr = torch.argsort(results_k[:, 0])
                    neighb = new_arr[results_k].T[None]

                    neighb = torch.cat(
                        (
                            neighb,
                            (Nmax - 1) * torch.ones((1, neighb.shape[1], Nmax - N)).cuda(),
                        ),
                        axis=2,
                    )


                    neigs.append(neighb)
                    upsample.append(torch.arange(pc.shape[0]).cuda(self.rank, non_blocking=True))
            else:
                for pc, pc_orig in zip(batch["pc"], batch["pc_orig"]):
                    N = pc.shape[0]

                    data = pc[:, :3].reshape(-1).astype(np.float32)
                    results_k = cp.zeros((N, 17), dtype=cp.int32)
                    results_1 = cp.zeros(pc_orig.shape[0], dtype=cp.int32)

                    results_k_ctypes = ctypes.cast(results_k.data.ptr, ctypes.POINTER(ctypes.c_int32))
                    results_1_ctypes = ctypes.cast(results_1.data.ptr, ctypes.POINTER(ctypes.c_int32))

                    full = pc_orig[:, :3].reshape(-1).astype(np.float32)

                    self.runTree_valid(self.valid_kdtree_obj, data, data.size, data, data.size, full, full.shape[0], results_k_ctypes, results_1_ctypes)
                    results_k = torch.from_dlpack(results_k.toDlpack()).long()
                    results_1 = torch.from_dlpack(results_1.toDlpack()).long()
                    
                    new_arr = torch.argsort(results_k[:, 0])
                    neighb = new_arr[results_k].T[None]

                    neighb = torch.cat(
                        (
                            neighb,
                            (Nmax - 1) * torch.ones((1, neighb.shape[1], Nmax - N)).cuda(),
                        ),
                        axis=2,
                    )
                    neigs.append(neighb)
                    upsample.append(new_arr[results_1.T])
                        
            neighbors_emb = torch.vstack(neigs).long()

            # Network inputs
            feat = batch["feat"].cuda(self.rank, non_blocking=True)
            labels = batch["labels_orig"].cuda(self.rank, non_blocking=True)
            cell_ind = batch["cell_ind"].cuda(self.rank, non_blocking=True)
            occupied_cell = batch["occupied_cells"].cuda(self.rank, non_blocking=True)
            net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

            # Get prediction and loss
            with torch.autocast("cuda", enabled=self.fp16):
                # Logits
                if training:
                    out = net(*net_inputs)
                else:
                    with torch.no_grad():
                        out = net(*net_inputs)
                # Upsample to original resolution
                out_upsample = []
                for id_b, closest_point in enumerate(upsample):
                    temp = out[id_b, :, closest_point]
                    out_upsample.append(temp.T)
                out = torch.cat(out_upsample, dim=0)
                # Loss
                loss = self.loss(out, labels)
            running_loss += loss.detach()

            # Confusion matrix
            with torch.no_grad():
                nb_class = out.shape[1]
                pred_label = out.max(1)[1]
                where = labels != 255
                confusion_matrix += fast_hist(
                    pred_label[where], labels[where], nb_class
                )

            # Logs
            if it % print_freq == print_freq - 1 or it == len(loader) - 1:
                # Gather scores
                if self.train_sampler is not None:
                    out = self.gather_scores([running_loss, confusion_matrix])
                else:
                    out = [running_loss.cpu(), confusion_matrix.cpu()]
                if self.rank == 0 or self.rank is None:
                    # Compute scores
                    oAcc = 100 * overall_accuracy(out[1])
                    mAcc = 100 * np.nanmean(per_class_accuracy(out[1]))
                    ious = per_class_iu(out[1])
                    mIoU = 100 * np.nanmean(ious)
                    running_loss_reduced = out[0].item() / self.world_size / (it + 1)
                    # Print score
                    self.print_log(running_loss_reduced, oAcc, mAcc, mIoU, ious)
                    # Save in tensorboard
                    if (writer is not None) and (training or it == len(loader) - 1):
                        header = "Train" if training else "Test"
                        step = (
                            self.current_epoch * len(loader) + it
                            if training
                            else self.current_epoch
                        )
                        writer.add_scalar(header + "/loss", running_loss_reduced, step)
                        writer.add_scalar(header + "/oAcc", oAcc, step)
                        writer.add_scalar(header + "/mAcc", mAcc, step)
                        writer.add_scalar(header + "/mIoU", mIoU, step)
                        writer.add_scalar(
                            header + "/lr", self.optim.param_groups[0]["lr"], step
                        )

            # Gradient step
            if training:
                self.optim.zero_grad(set_to_none=True)
                if self.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        # Return score
        if self.rank == 0 or self.rank is None:
            return mIoU
        else:
            return None

    def load_state(self, best=False):
        filename = self.path_to_ckpt
        # filename += "/ckpt_best.pth" if best else "/ckpt_last.pth"
        rank = 0 if self.rank is None else self.rank
        ckpt = torch.load(
            filename,
            map_location=f"cuda:{rank}",
        )
        self.net.load_state_dict(ckpt["net"])
        # if ckpt.get("optim") is None:
        #     warnings.warn("Optimizer state not available")
        # else:
        #     self.optim.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            if ckpt.get("scheduler") is None:
                warnings.warn("Scheduler state not available")
            else:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.fp16:
            if ckpt.get("scaler") is None:
                warnings.warn("Scaler state not available")
            else:
                self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("best_miou") is not None:
            self.best_miou = ckpt["best_miou"]
        if ckpt.get("epoch") is not None:
            self.current_epoch = ckpt["epoch"] + 1
        print(
            f"Checkpoint loaded on {torch.device(rank)} (cuda:{rank}): {self.path_to_ckpt}"
        )

    def save_state(self, best=False):
        if self.rank == 0 or self.rank is None:
            dict_to_save = {
                "epoch": self.current_epoch,
                "net": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "scaler": self.scaler.state_dict() if self.fp16 else None,
                "best_miou": self.best_miou,
            }
            filename = self.path_to_ckpt
            filename += "/ckpt_best.pth" if best else "/ckpt_last.pth"
            torch.save(dict_to_save, filename)

    def train(self):
        for _ in range(self.current_epoch, self.max_epoch):
            # Train
            self.one_epoch(training=True)

            if self.current_epoch>5:
                # Val
                miou = self.one_epoch(training=False)
                # Save best checkpoint
                if miou is not None and miou > self.best_miou:
                    self.best_miou = miou
                    self.save_state(best=True)
                    print(f"\n\n*** New best mIoU: {self.best_miou:.1f}.\n")
                # Save last checkpoint
                self.save_state()
            # Increase epoch number
            self.current_epoch += 1
        if self.rank == 0 or self.rank is None:
            print("Finished Training")
