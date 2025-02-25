# python imports
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.optim import Adam
# internal imports
from torch.utils.data import DataLoader

from Model import losses
from Model.config import Config as args
from Model.datagenerators import OASISBrainDataset, OASISBrainInferDataset
from Model.model import SpatialTransformer
from Model.model import U_Network as U_Network
import warnings
from Model import trans
from torchvision import transforms
import glob

warnings.filterwarnings('ignore')


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train():
    model_dir = f'{args.model_dir}{args.sim_loss}_{args.edge_sim_loss}_{args.alpha}_{args.beta}_{args.input_channel}'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # 创建需要的文件夹并指定gpu
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    train_set = OASISBrainDataset(glob.glob(args.train_dir + '*.pkl'), transforms=train_composed)
    val_set = OASISBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

    vol_size = train_set[0][0].shape[1:]

    print("Number of training images: ", len(train_set))
    print(f'image shape: {vol_size}')

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode='nearest').to(device)
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # Set optimizer and losses
    opt = Adam(UNet.parameters(), lr=args.lr)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    edge_sim_loss_fn = losses.ncc_loss if args.edge_sim_loss == 'ncc' else losses.mse_loss
    dice_fn = losses.dice_oasis
    grad_loss_fn = losses.gradient_loss
    max_dice = 0

    # Training loop.
    for i in range(args.max_epochs):
        UNet.eval()
        with torch.no_grad():
            valid_dice_list = []
            jac_list = []
            for valid_iter_, valid_d in enumerate(val_loader):
                m, f, ml, fl, me, fe = valid_d

                # [B, C, D, W, H]
                moving = m.to(device).float()
                fixed = f.to(device).float()
                ml = ml.to(device).float()
                fl = fl.to(device).float()
                me = me.to(device).float()
                fe = fe.to(device).float()

                # Run the data through the model to produce warp and flow field
                flow_m2f = UNet(moving, fixed, me, fe)
                label_m2f = STN_label(ml, flow_m2f)

                dice = dice_fn(label_m2f, fl)
                valid_dice_list.append(dice.item())

                tar = moving.detach().cpu().numpy()[0, 0, ...]
                jac_den = np.prod(tar.shape)
                for flow_item in flow_m2f:
                    jac_det = losses.jacobian_determinant(flow_item.detach().cpu().numpy())
                    jac_list.append(np.sum(jac_det <= 0) / jac_den)

            valid_mean_dice = np.array(valid_dice_list).mean()
            if i == 0:
                print(
                    f'start -- dice: {valid_mean_dice}, jacob_mean:{np.array(jac_list).mean():.7f}, jacob_std:{np.array(jac_list).std():.7f}')
            else:
                print(
                    f'epoch: {i}/{args.max_epochs + 1}, current dice: {valid_mean_dice}, jacob_mean:{np.array(jac_list).mean():.7f},'
                    f'jacob_std:{np.array(jac_list).std():.7f}, best_dice: {max_dice}')

            if valid_mean_dice >= max_dice:
                max_dice = valid_mean_dice
                # Save model checkpoint
                saved_files = os.listdir(model_dir)
                if len(saved_files) >= args.save_model_num:
                    saved_files.sort()
                    last_one = saved_files[0]
                    os.remove(os.path.join(model_dir, last_one))
                save_file_name = os.path.join(model_dir, f'dice{valid_mean_dice:.5f}_jacob{np.array(jac_list).mean():.5f}.pth')
                torch.save(UNet.state_dict(), save_file_name)

        UNet.train()

        for iter_, d in enumerate(train_loader):
            m, f, ml, fl, me, fe = d
            # [B, C, D, W, H]

            input_moving = m.to(device).float()
            input_fixed = f.to(device).float()
            input_me = me.to(device).float()
            input_fe = fe.to(device).float()

            flow_m2f = UNet(input_moving, input_fixed, input_me, input_fe)
            m2f = STN(input_moving, flow_m2f)
            edge_m2f = STN(input_me, flow_m2f)

            image_sim_loss = sim_loss_fn(m2f, input_fixed)
            edge_sim_loss = edge_sim_loss_fn(edge_m2f, input_fe)

            grad_loss = grad_loss_fn(flow_m2f)
            sim_loss = image_sim_loss + args.beta * edge_sim_loss

            loss = sim_loss + args.alpha * grad_loss

            opt.zero_grad()
            loss.backward()
            opt.step()


if __name__ == "__main__":
    train()
