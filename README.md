# WaffleIron

![](./Network.png)
![](./Block.png)

[**Exploiting Local Features and Range Images for Small Data Real-Time Point Cloud Semantic Segmentation**](https://theresebeharrie.com/2018/07/09/three-ways-deal-waiting-publishing/)  
[*Daniel Fusaro*<sup>1</sup>](https://www.dei.unipd.it/en/persona/85ed20c7cab60b5a1ee989237cc70ec2),
[*Simone Mosco*<sup>1</sup>](https://www.dei.unipd.it/persona/ec7b2218b236d88ae0be5ca08e73ab80),
[*Alberto Pretto*<sup>1,2</sup>](https://www.dei.unipd.it/en/persona/C36EC9D29C2C5DFB03BFE9E045B32FD9)  
<sup>1</sup>*Department of Information Engineering, University of Padova, Italy.*

If you find this code or work useful, please cite the following [paper](http://arxiv.org/abs/2301.10100):
```
@inproceedings{puy23waffleiron,
  title={Using a Waffle Iron for Automotive Point Cloud Semantic Segmentation},
  author={Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle={ICCV},
  year={2023}
}

// our paper is pending for acceptance at IROS 2024
```

<!--## Updates

This work was accepted at ICCV23. The code and trained models were updated on September 21, 2023 to reproduce the scores in the published version. 

If you need to access the preliminary trained models you can refer to this [section](#Preliminary-version). Note that those preliminary models are not performing as well as those used in the published version.-->


## Installation

Setup the environment and clone this repo:
```
pip install pyaml==6.0 tqdm=4.63.0 scipy==1.8.0 torch==1.11.0 tensorboard=2.8.0
pip3 install pycuda pycu
git clone https://github.com/Bender97/WaffleAndRange
pip install -e ./
```

Then, compile the cuda related stuff.
```
cd cudastuff
mkdir build && cd build
cmake ..
make -j5
```


Download the trained models:
```
wget https://github.com/valeoai/WaffleIron/files/10294733/info_datasets.tar.gz
tar -xvzf info_datasets.tar.gz
```
[**SemanticKITTI model**](https://drive.google.com/file/d/1bSiQIvdA9P08NJS05qNXpcsAd6l_sIDw/view?usp=sharing)

<!--wget https://github.com/valeoai/WaffleIron/releases/download/v0.2.0/waffleiron_nuscenes.tar.gz
tar -xvzf waffleiron_nuscenes.tar.gz-->

If you want to uninstall this package, type `pip uninstall waffleiron`.


## Testing pretrained models

### Option 1: Using this code

To evaluate the nuScenes trained model, type
```
python launch_train.py \
--dataset nuscenes \
--path_dataset /path/to/nuscenes/ \
--log_path ./pretrained_models/WaffleIron-48-384__nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--fp16 \
--restart \
--eval
```
This should give you a final mIoU of 77.6%.

To evaluate the SemanticKITTI trained model, type
```
python launch_train.py \
--dataset semantic_kitti \
--path_dataset /path/to/kitti/ \
--log_path ./pretrained_models/WaffleIron-48-256__kitti/ \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--fp16 \
--restart \
--eval
```
This should give you a final mIoU of 68.0%.

**Remark:** *On SemanticKITTI, the code above will extract object instances on the train set (despite this being not necessary for validation) because this augmentation is activated for training on this dataset (and this code re-use the training script). This can be bypassed by editing the `yaml` config file and changing the entry `instance_cutmix` to `False`. The instances are saved automatically in `/tmp/semantic_kitti_instances/`.*

### Option 2: Using the official APIs

The second option writes the predictions on disk and the results can be computed using the official nuScenes or SemanticKITTI APIs. This option also allows you to perform test time augmentations, which is not possible with Option 1 above. These scripts should be useable for submission of the official benchmarks.

#### nuScenes

To extract the prediction with the model trained on nuScenes, type
```
python eval_nuscenes.py \
--path_dataset /path/to/nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--ckpt ./pretrained_models/WaffleIron-48-384__nuscenes/ckpt_last.pth \
--result_folder ./predictions_nuscenes \
--phase val \
--num_workers 12
```
or, if you want to use, e.g., 10 votes with test time augmentations,
```
python eval_nuscenes.py \
--path_dataset /path/to/nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--ckpt ./pretrained_models/WaffleIron-48-384__nuscenes/ckpt_last.pth \
--result_folder ./predictions_nuscenes \
--phase val \
--num_workers 12 \
--num_votes 10 \
--batch_size 10
```
You can reduce `batch_size` to 5, 2 or 1 depending on the available memory.

These predictions can be evaluated using the official nuScenes API as follows
```
git clone https://github.com/nutonomy/nuscenes-devkit.git
python nuscenes-devkit/python-sdk/nuscenes/eval/lidarseg/evaluate.py \
--result_path ./predictions_nuscenes \
--eval_set val \
--version v1.0-trainval \
--dataroot /path/to/nuscenes/ \
--verbose True  
```

#### SemanticKITTI

To extract the prediction with the model trained on SemanticKITTI, type
```
python eval_kitti.py \
--path_dataset /path/to/kitti/ \
--ckpt ./pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--result_folder ./predictions_kitti \
--phase val \
--num_workers 12
```

The predictions can be evaluated using the official APIs by typing
```
git clone https://github.com/PRBonn/semantic-kitti-api.git
cd semantic-kitti-api/
python evaluate_semantics.py \
--dataset /path/to/kitti//dataset \
--predictions ../predictions_kitti \
--split valid
```

## Training

### nuScenes

To retrain the WaffleIron-48-384 backbone on nuScenes type
```
python launch_train.py \
--dataset nuscenes \
--path_dataset /path/to/nuscenes/ \
--log_path ./logs/WaffleIron-48-384__nuscenes/ \
--config ./configs/WaffleIron-48-384__nuscenes.yaml \
--multiprocessing-distributed \
--fp16
```

We used the checkpoint at the *last* training epoch to report the results.

Note: for single-GPU training, you can remove `--multiprocessing-distributed` and add the argument `--gpu 0`.


### SemanticKITTI

To retrain the WaffleIron-48-256 backbone, type
```
python launch_train.py \
--dataset semantic_kitti \
--path_dataset /path/to/kitti/ \
--log_path ./logs/WaffleIron-48-256__kitti \
--config ./configs/WaffleIron-48-256__kitti.yaml \
--multiprocessing-distributed \
--fp16
```

At the beginning of the training, the instances for cutmix augmentation are saved in `/tmp/semantic_kitti_instances/`. If this process is interrupted before completion, please delete `/tmp/semantic_kitti_instances/` and relaunch training. You can disable the instance cutmix augmentations by editing the `yaml` config file to set `instance_cutmix` to `False`.

For submission to the official benchmark on the test set of SemanticKITTI, we trained the network on both the val and train sets (argument `--trainval` in `launch_train.py`), used the checkpoint at the last epoch and 12 test time augmentations during inference.


## Creating your own network

### Models

The WaffleIron backbone is defined in `waffleiron/backbone.py` and can be imported in your project by typing
```python
from waffleiron import WaffleIron
```
It needs to be combined with a embedding layer to provide point tokens and a pointwise classification layer, as we do in `waffleiron/segmenter.py`. You can define your own embedding and classification layers instead.


## Preliminary version

To access the preliminary trained models and the corresponding code, you can clone version v0.1.1 of the code.
```
git clone -b v0.1.1 https://github.com/valeoai/WaffleIron
cd WaffleIron/
pip install -e ./
```

The corresponding pretrained models are available at:
```
wget https://github.com/valeoai/WaffleIron/files/10294734/pretrained_nuscenes.tar.gz
tar -xvzf pretrained_nuscenes.tar.gz
wget https://github.com/valeoai/WaffleIron/files/10294735/pretrained_kitti.tar.gz
tar -xvzf pretrained_kitti.tar.gz
```

## Acknowledgements

We thank the author of
https://github.com/ingowald/cudaKDTree

for making their [implementation](https://github.com/ingowald/cudaKDTree) of the KDTree publicly available and very easy to use and understand.



## License
WaffleAndRange is released under the [Apache 2.0 license](./LICENSE). 

The implementation of the Lovász loss in `utils/lovasz.py` is released under 
[MIT Licence](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/LICENSE).
