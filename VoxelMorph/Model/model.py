import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Model.config import Config as args


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.input_channel = args.input_channel
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = self.input_channel if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 3, 1, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + self.input_channel, dec_nf[5], batchnorm=bn))
        self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(2)
        self.conv_1x1 = self.conv_block(dim, enc_nf[-1], enc_nf[-1], kernel_size=1, padding=0)
        self.feature_layers1 = self.feature_block1(dim)
        self.feature_layers2 = self.feature_block2(dim)
        self.feature_layers3 = self.feature_block3(dim)
        self.max_pool = getattr(nn, "MaxPool{0}d".format(dim))(2)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def feature_block1(self, dim):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        layer1 = nn.Sequential(
            conv_fn(2, args.input_channel // 2, 3, 1, 1),
            bn_fn(args.input_channel // 2),
            nn.LeakyReLU(0.2),
        )
        return layer1

    def feature_block2(self, dim):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        layer2 = nn.Sequential(
            conv_fn(args.input_channel // 2, args.input_channel, 3, 1, 1),
            bn_fn(args.input_channel),
            nn.LeakyReLU(0.2),

            conv_fn(args.input_channel, args.input_channel // 2, 3, 1, 1),
            bn_fn(args.input_channel // 2),
            nn.LeakyReLU(0.2),

            up_sample,
        )
        return layer2

    def feature_block3(self, dim):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        layer3 = nn.Sequential(
            conv_fn(args.input_channel, args.input_channel // 2, 3, 1, 1),
            bn_fn(args.input_channel // 2),
            nn.LeakyReLU(0.2),
        )
        return layer3

    def forward(self, src, tgt, src_edge, tgt_edge):
        if args.input_channel > 2:
            src_set = torch.cat((src, src_edge), dim=1)
            tgt_set = torch.cat((tgt, tgt_edge), dim=1)

            src_feature1 = self.feature_layers1(src_set)
            src_feature2 = self.feature_layers2(self.max_pool(src_feature1))
            src_feature = self.feature_layers3(torch.cat((src_feature1, src_feature2), dim=1))

            tgt_feature1 = self.feature_layers1(tgt_set)
            tgt_feature2 = self.feature_layers2(self.max_pool(tgt_feature1))
            tgt_feature = self.feature_layers3(torch.cat((tgt_feature1, tgt_feature2), dim=1))

            x = torch.cat([src_feature, tgt_feature], dim=1)
        else:
            x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        max_pool = getattr(nn, "MaxPool{0}d".format(self.dim))
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x = max_pool(2)(x)
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = self.conv_1x1(x_enc[-1])
        for i in range(3):
            y = self.dec[i](y)
            # if i != 0:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
