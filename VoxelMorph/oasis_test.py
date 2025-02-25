import torch
import numpy as np
# internal imports
from torch.utils.data import DataLoader

import hausdorff
from Model import losses
from Model.config import Config as args
from Model.datagenerators import OASISBrainInferDataset
from Model.model import SpatialTransformer
from Model.model import U_Network as U_Network
import warnings
from Model import trans
from torchvision import transforms
import glob
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def test():
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    test_set = OASISBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    vol_size = test_set[0][1].shape[1:]

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode='nearest').to(device)
    STN.train()
    dice_fn = losses.dice_oasis
    best_UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    best_UNet.load_state_dict(torch.load(args.checkpoint_path, map_location=f'cuda:{args.gpu}'))
    best_UNet.eval()
    with torch.no_grad():
        dice_list = []
        jac_list = []
        hd_list = []
        hd_95_list = []
        for test_iter_, test_d in enumerate(test_loader):
            m, f, ml, fl, me, fe = test_d
            # [B, C, D, W, H]
            moving_label = ml.to(device).float()
            fixed_label = fl.to(device).float()
            moving = m.to(device).float()
            fixed = f.to(device).float()
            moving_edge = me.to(device).float()
            fixed_edge = fe.to(device).float()

            # Run the data through the model to produce warp and flow field

            flow_m2f = best_UNet(moving, fixed, moving_edge, fixed_edge)
            m2f_label = STN_label(moving_label, flow_m2f)

            hd, hd95 = hausdorff.calculate_hd(m2f_label.detach().cpu().numpy(), fixed_label.detach().cpu().numpy())
            hd_list.append(hd)
            hd_95_list.append(hd95)
            # Calculate dice score
            dice_score = dice_fn(m2f_label, fixed_label)
            dice_list.append(dice_score.item())

            tar = moving.detach().cpu().numpy()[0, 0, ...]
            jac_den = np.prod(tar.shape)
            for flow_item in flow_m2f:
                jac_det = losses.jacobian_determinant(flow_item.detach().cpu().numpy())
                jac_list.append(np.sum(jac_det <= 0) / jac_den)
            print(test_iter_, dice_score.item(), np.sum(jac_det <= 0) / jac_den, hd, hd95)

        mean_dice = np.array(dice_list).mean()
    print(
        f'dice:{mean_dice:.5f},jacob_mean:{np.array(jac_list).mean():.7f},jacob_std:{np.array(jac_list).std():.7f},'
        f'hd: {np.mean(hd_list):.7f}, hd95: {np.mean(hd_95_list):.7f}')


test()
