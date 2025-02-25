# python imports
import os
import glob
import warnings
# external imports
import numpy
import torch
import numpy as np
import torch.utils.data as Data
# internal imports
from medpy.metric.binary import __surface_distances

from utils import losses
from utils.config import args
from utils.datagenerators import OASISBrainInferDataset
from Models.STN import SpatialTransformer

warnings.filterwarnings('ignore')

from Models.TransMatch import TransMatch


def hd_func(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`asd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def hd95_func(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95


def calculate_hd(ml, fl):
    masks = np.unique(np.concatenate((ml, fl), 0))
    hd_list = []
    hd_95_list = []
    for k in masks:
        if k == 0:
            continue
        hd95 = hd95_func(ml == k, fl == k)
        hd_list.append(hd95)
        hd_95_list.append(hd95)
    return np.mean(hd_list), np.mean(hd_95_list)


def infer():
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    model_path = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha) + "_" + str(args.beta) + "_" + str(
        args.input_channel) + "_" + args.edge_sim_loss
    test_DS = OASISBrainInferDataset(glob.glob(args.test_dir + '*.pkl'))
    test_DL = Data.DataLoader(test_DS, batch_size=1, shuffle=False, pin_memory=True)
    print("Number of test images: ", len(test_DS))

    vol_size = test_DS[0][0].shape[1:]
    print("vol size: ", vol_size)
    # 创建配准网络（net）和STN
    net = TransMatch(args).to(device)

    model_dir = args.model_dir + model_path
    saved_models = os.listdir(model_dir)
    saved_models.sort()
    best_model = saved_models[-1]
    checkpoint = torch.load(args.model_dir + model_path + '/' + best_model)['state_dict']
    net.load_state_dict(checkpoint)

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

    net.eval()
    STN.eval()
    DSC = []
    JAC = []
    HD95 = []
    with torch.no_grad():
        for test_d in test_DL:
            moving, fixed, moving_label, fixed_label, moving_edge, fixed_edge = test_d
            # 读入moving图像
            moving = moving.to(device).float()
            fixed = fixed.to(device).float()
            moving_label = moving_label.to(device).float()
            fixed_label = fixed_label.to(device).float()
            moving_edge = moving_edge.to(device).float()
            fixed_edge = fixed_edge.to(device).float()
            # 获得配准后的图像和label
            pred_flow = net(moving, fixed, moving_edge, fixed_edge)
            pred_label = STN_label(moving_label, pred_flow)

            # 计算DSC
            dice = losses.dice_oasis(fixed_label, pred_label)
            DSC.append(dice)
            tar = moving.detach().cpu().numpy()[0, 0, ...]
            jac_den = np.prod(tar.shape)
            for flow_item in pred_flow:
                jac_det = losses.jacobian_determinant(flow_item.detach().cpu().numpy())
                JAC.append(np.sum(jac_det <= 0) / jac_den)
            _, hd95 = calculate_hd(fixed_label.detach().cpu().numpy(), pred_label.detach().cpu().numpy())
            HD95.append(hd95)
            del moving, fixed, moving_label, fixed_label, moving_edge, fixed_edge, pred_flow, pred_label

        print(
            f'mean dice: {np.mean(DSC)}, std dice: {np.std(DSC)}, mean jac: {np.mean(JAC)}, std jac: {np.std(JAC)}, hd95: {np.mean(HD95)}')


if __name__ == "__main__":
    infer()
