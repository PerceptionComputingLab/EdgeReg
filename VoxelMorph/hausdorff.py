import numpy
import numpy as np
from Model.datagenerators import OASISBrainInferDataset
from medpy.metric.binary import hd, hd95, __surface_distances
import glob
from Model import trans
from Model.config import Config as args
from torchvision import transforms
import SimpleITK as sitk
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer


def one_class_hausdorff_distance(y_pred, y, spacing=(1., 1., 1.)):
    """
    Hausdorff distance between two label masks (not one-hot) for one class

    Args:
        y_pred: (numpy.ndarray, shape (N, *sizes)) binary predicted label mask
        y: (numpy.ndarray, shape (N, *sizes)) binary ground truth label mask
        spacing (list, float): pixel/voxel spacings

    Returns:
        hausdorff_distance (float)
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    batch_size = y_pred.shape[0]
    result = []

    for i in range(batch_size):
        y_pred_img = sitk.GetImageFromArray(y_pred[i].astype('float32'))
        y_pred_img.SetSpacing(spacing)
        y_img = sitk.GetImageFromArray(y[i].astype('float32'))
        y_img.SetSpacing(spacing)
        try:
            hausdorff_distance_filter.Execute(y_pred_img, y_img)
            hd = hausdorff_distance_filter.GetHausdorffDistance()
            result.append(hd)
        except:
            # skip empty masks
            if y_pred[i].sum() == 0 or y[i].sum() == 0:
                continue
    return np.mean(result)


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
    return hd_list, hd_95_list


def test():
    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_set = OASISBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), transforms=test_composed)
    hausdorff = []
    hausdorff95 = []
    # metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95')]
    # labels = {i: str(i) for i in range(1, 36)}
    # evaluator = eval_.SegmentationEvaluator(metrics, labels)
    for i, data in enumerate(test_set):
        # print(f'Evaluating {i + 1}...')
        _, _, ml, fl, _, _ = data
        # evaluator.evaluate(ml.numpy()[0], fl.numpy()[0], str(i))
        hd_v, hd95_v = calculate_hd(ml.numpy(), fl.numpy())
        hausdorff.extend(hd_v)
        hausdorff95.extend(hd95_v)
        print(hd_v, hd95_v)
    print(f'hd: {np.mean(hausdorff)}, hd95: {np.mean(hausdorff95)}')
    # print('\nSubject-wise results...')
    # writer.ConsoleWriter().write(evaluator.results)
    # writer.CSVWriter('./hd95result.csv').write(evaluator.results)


test()
