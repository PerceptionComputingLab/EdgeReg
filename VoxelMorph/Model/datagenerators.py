from torch.utils.data import Dataset
import numpy as np
import pickle
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import torch


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class OASISBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.sobel_operator = sitk.SobelEdgeDetectionImageFilter()

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x_edge, y_edge = self.sobel_operator.Execute(sitk.GetImageFromArray(x[0])), self.sobel_operator.Execute(
            sitk.GetImageFromArray(y[0]))
        x_edge, y_edge = sitk.GetArrayFromImage(x_edge), sitk.GetArrayFromImage(y_edge)
        x_edge, y_edge = x_edge[None, ...], y_edge[None, ...]
        x_edge, y_edge = x_edge / np.max(x_edge), y_edge / np.max(y_edge)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x_edge = np.ascontiguousarray(x_edge)
        y_edge = np.ascontiguousarray(y_edge)
        x, y, x_seg, y_seg, x_edge, y_edge = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(
            x_seg), torch.from_numpy(y_seg), torch.from_numpy(x_edge), torch.from_numpy(y_edge)
        return x, y, x_seg, y_seg, x_edge, y_edge

    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        self.sobel_operator = sitk.SobelEdgeDetectionImageFilter()

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x_edge, y_edge = self.sobel_operator.Execute(sitk.GetImageFromArray(x[0])), self.sobel_operator.Execute(
            sitk.GetImageFromArray(y[0]))
        x_edge, y_edge = sitk.GetArrayFromImage(x_edge), sitk.GetArrayFromImage(y_edge)
        x_edge, y_edge = x_edge[None, ...], y_edge[None, ...]
        x_edge, y_edge = x_edge / np.max(x_edge), y_edge / np.max(y_edge)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x_edge = np.ascontiguousarray(x_edge)
        y_edge = np.ascontiguousarray(y_edge)
        x, y, x_seg, y_seg, x_edge, y_edge = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(
            x_seg), torch.from_numpy(y_seg), torch.from_numpy(x_edge), torch.from_numpy(y_edge)
        return x, y, x_seg, y_seg, x_edge, y_edge

    def __len__(self):
        return len(self.paths)
