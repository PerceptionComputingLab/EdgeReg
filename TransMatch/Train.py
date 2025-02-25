# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators import OASISBrainDataset, OASISBrainInferDataset
from Models.STN import SpatialTransformer
from natsort import natsorted
from torchvision import transforms
warnings.filterwarnings('ignore')

from Models.TransMatch import TransMatch


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.model_dir + '/' + str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha) + "_" + str(args.beta) + "_" + str(args.input_channel) + "_" + args.edge_sim_loss):
        os.makedirs(args.model_dir + '/' + str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha) + "_" + str(args.beta) + "_" + str(args.input_channel) + "_" + args.edge_sim_loss)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def save_checkpoint(state, save_dir=args.model_dir, filename='checkpoint.pth.tar', max_model_num=3):
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) >= max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, save_dir + filename)


def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha) + "_" + str(args.beta) + "_" + str(args.input_channel) + "_" + args.edge_sim_loss
    print("log_name: ", log_name)
    log_f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # Get all the names of the training data
    DS = OASISBrainDataset(glob.glob(args.train_dir + '*.pkl'))
    test_DS = OASISBrainInferDataset(glob.glob(args.test_dir +'*.pkl'))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_DL = Data.DataLoader(test_DS, batch_size=1, shuffle=False, pin_memory=True)
    print("Number of training images: ", len(DS))

    vol_size = DS[0][0].shape[1:]
    print("vol size: ", vol_size)
    # 创建配准网络（net）和STN
    net = TransMatch(args).to(device)

    iterEpoch = 1
    contTrain = True
    if contTrain:
        model_dir = args.model_dir + log_name
        saved_models = os.listdir(model_dir)
        saved_models.sort()
        best_model = saved_models[-1]
        checkpoint = torch.load(args.model_dir + log_name + '/' + best_model)['state_dict']
        net.load_state_dict(checkpoint)
        iterEpoch = 201

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    # UNet.train()
    net.train()
    STN.train()

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    edge_sim_loss_fn = losses.ncc_loss if args.edge_sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    best_model = [0, 0, 0, 0, 0]

    # Training loop.
    for i in range(iterEpoch, args.n_iter + 1000):
        # Generate the moving images and convert them to tensors.
        net.train()
        STN.train()
        print('epoch:', i)
        for d in DL:
            moving, fixed, _, _, moving_edge, fixed_edge = d
            moving = moving.to(device).float()
            fixed = fixed.to(device).float()
            moving_edge = moving_edge.to(device).float()
            fixed_edge = fixed_edge.to(device).float()

            del _
                
            # [B, C, D, W, H]
            # Run the data through the model to produce warp and flow field

            flow_m2f = net(moving, fixed, moving_edge, fixed_edge)
            m2f = STN(moving, flow_m2f)
            m2f_edge = STN(moving_edge, flow_m2f)

            # Calculate loss
            image_sim_loss = sim_loss_fn(m2f, fixed)
            edge_sim_loss = losses.mse_loss(m2f_edge, fixed_edge)
            sim_loss = image_sim_loss + args.beta * edge_sim_loss
            grad_loss = grad_loss_fn(flow_m2f)
            # zero_loss = zero_loss_fn(flow_m2f, zero)
            loss = sim_loss + args.alpha * grad_loss  # + zero_loss

            print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=log_f)
            del flow_m2f, m2f, m2f_edge
            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            # inverse fixed image and moving image
            flow_f2m = net(fixed, moving, fixed_edge, moving_edge)
            f2m = STN(fixed, flow_f2m)
            edge_f2m = STN(fixed_edge, flow_f2m)

            # Calculate loss
            image_sim_loss = sim_loss_fn(f2m, moving)
            edge_sim_loss = edge_sim_loss_fn(edge_f2m, moving_edge)
            sim_loss = image_sim_loss + args.beta * edge_sim_loss
            grad_loss = grad_loss_fn(flow_f2m)
            loss = sim_loss + args.alpha * grad_loss  # + zero_loss

            print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=log_f)
            del flow_f2m, f2m, edge_f2m, moving, fixed, moving_edge, fixed_edge
            # Backwards and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        STN.eval()
        STN_label.eval()

        DSC = []
        JAC = []
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
                del moving, fixed, moving_label, fixed_label, moving_edge, fixed_edge, pred_flow, pred_label
            current_dice = np.mean(DSC)
            current_dice_std = np.std(DSC)
            current_jacob = np.mean(JAC)
            current_jacob_std = np.std(JAC)
            if current_dice >= best_model[1]:
                best_model = [i, current_dice, current_dice_std, current_jacob, current_jacob_std]
            print(f'mean dice: {np.mean(DSC)}, std dice: {np.std(DSC)}, mean jac: {np.mean(JAC)}, std jac: {np.std(JAC)}, {best_model}')
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': net.state_dict(),
            'optimizer': opt.state_dict(),
        }, save_dir = args.model_dir + log_name + '/', filename='dsc{:.4f}epoch{:0>3d}.pth.tar'.format(np.mean(DSC), i + 1))

    log_f.close()


if __name__ == "__main__":
    train()
