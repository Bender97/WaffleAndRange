import ctypes
from ctypes import cdll
lib = cdll.LoadLibrary('build/libmylib.so')

import numpy as np
from numpy.ctypeslib import ndpointer

import time
from scipy.spatial import cKDTree as KDTree


class MyFastKDTree(object):
    def __init__(self):
        self.obj = lib.MyFastKDTree_new()
        self.fun = lib.MyFastKDTree_call
        
        self.fun.argtypes = [ctypes.c_void_p,
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t]

    def call(self, idx):
        
        data = np.fromfile(f"/media/fusy/Windows-SSD/Users/orasu/Desktop/SemanticKittiDataset/data_odometry_velodyne/dataset/sequences/00/velodyne/{str(idx).zfill(6)}.bin", dtype=np.float32).reshape((-1, 4))
        pc = np.fromfile(f"/media/fusy/Windows-SSD/Users/orasu/Desktop/SemanticKittiDataset/data_odometry_velodyne/dataset/sequences/00/velodyne/{str(idx).zfill(6)}.bin", dtype=np.float32).reshape((-1, 4))[:, :3]
        kdtree = KDTree(data[:, :3])


        N = data.shape[0]

        print(data.shape)
        _, neighbors_emb = kdtree.query(data[:, :3], k=16 + 1)
        #print(data[:10, :3])
        #print(neighbors_emb[:10, :])
        print("neig", data[neighbors_emb[0, 0], :])
        print("neig", data[neighbors_emb[15, 0], :])

        data = data[:, :3].reshape(-1).astype(np.float32)

        self.fun.restype = ndpointer(ctypes.c_int, shape=(17, N))
        s = time.time()
        a = self.fun(self.obj, data, data.size)

        new_arr = np.argsort(a[0, :])
        a2 = np.array(a.T)[new_arr]

        e = time.time()
        print("time", e-s)

        idx = a2[15, 0]
        print("reco", pc[idx, :])



s = time.time()
f = MyFastKDTree()
e = time.time()
print("creation time", e-s)
for i in range(10):
    f.call(i)

