from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .pytorch_resnet_v2_50 import KitModel
from .v2_93 import *
import numpy as np
from PIL import Image
import cv2
import time
import os
import math

def transformer(U, theta, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """
    def _repeat(x, n_repeats):
        rep = torch._cast_Long(torch.transpose(torch.ones([n_repeats, ]).unsqueeze(1), 1, 0))
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)

    def _interpolate(im, x, y, out_size):
        num_batch, height, width, channels = im.size() # to be sure the input dims is NHWC
        x = torch._cast_Float(x).cuda()
        y = torch._cast_Float(y).cuda()
        height_f = torch._cast_Float(torch.Tensor([height]))[0].cuda()
        width_f = torch._cast_Float(torch.Tensor([width]))[0].cuda()
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([], dtype=torch.int32).cuda()
        max_y = torch._cast_Long(torch.Tensor([height - 1]))[0].cuda()
        max_x = torch._cast_Long(torch.Tensor([width - 1]))[0].cuda()

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0

        # do sampling
        x0 = torch._cast_Long(torch.floor(x)).cuda()
        x1 = x0 + 1
        y0 = torch._cast_Long(torch.floor(y)).cuda()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to look up pixels in the flate images
        # and restore channels dim
        im_flat = im.contiguous().view(-1, channels)
        im_flat = torch._cast_Float(im_flat)
        Ia = im_flat[idx_a] # as in tf, the default dim is row first
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # calculate interpolated values
        x0_f = torch._cast_Float(x0).cuda()
        x1_f = torch._cast_Float(x1).cuda()
        y0_f = torch._cast_Float(y0).cuda()
        y1_f = torch._cast_Float(y1).cuda()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)
        
        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def _meshgrid(height, width):
        x_t = torch.matmul(torch.ones([height, 1]), 
                            torch.transpose(torch.linspace(-1.0, 1.0, width).unsqueeze(1), 1, 0))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).unsqueeze(1),
                            torch.ones([1, width]))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def pinv(A):
        A = A.cpu() + torch.eye(8) * 1e-4
        return torch.inverse(A).cuda()

    def get_H(ori, tar):
        num_batch = ori.size()[0]
        one = torch.ones([num_batch, 1]).cuda()
        zero = torch.zeros([num_batch, 1]).cuda()
        x = [ori[:, 0:1], ori[:, 2:3], ori[:, 4:5], ori[:, 6:7]]
        y = [ori[:, 1:2], ori[:, 3:4], ori[:, 5:6], ori[:, 7:8]]
        u = [tar[:, 0:1], tar[:, 2:3], tar[:, 4:5], tar[:, 6:7]]
        v = [tar[:, 1:2], tar[:, 3:4], tar[:, 5:6], tar[:, 7:8]]

        A_ = []
        A_.extend([x[0], y[0], one, zero, zero, zero, -x[0] * u[0], -y[0] * u[0]])
        A_.extend([x[1], y[1], one, zero, zero, zero, -x[1] * u[1], -y[1] * u[1]])
        A_.extend([x[2], y[2], one, zero, zero, zero, -x[2] * u[2], -y[2] * u[2]])
        A_.extend([x[3], y[3], one, zero, zero, zero, -x[3] * u[3], -y[3] * u[3]])
        A_.extend([zero, zero, zero, x[0], y[0], one, -x[0] * v[0], -y[0] * v[0]])
        A_.extend([zero, zero, zero, x[1], y[1], one, -x[1] * v[1], -y[1] * v[1]])
        A_.extend([zero, zero, zero, x[2], y[2], one, -x[2] * v[2], -y[2] * v[2]])
        A_.extend([zero, zero, zero, x[3], y[3], one, -x[3] * v[3], -y[3] * v[3]])
        A = torch.cat(A_, dim=1).view(num_batch, 8, 8)
        b_ = [u[0], u[1], u[2], u[3], v[0],v[1], v[2], v[3]]
        b = torch.cat(b_, dim=1).view([num_batch, 8, 1])

        ans = torch.cat([torch.matmul(pinv(A), b).view([num_batch, 8]), torch.ones([num_batch, 1]).cuda()], 
                        dim=1)
        return ans

    # check pass
    def get_Hs(theta):
        num_batch = theta.size()[0]
        h = 2.0 / grid_h
        w = 2.0 / grid_w
        Hs = []
        for i in range(grid_h):
            for j in range(grid_w):
                hh = i * h - 1
                ww = j * w - 1
                ori = torch._cast_Float(torch.Tensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h])).\
                    view([1, 8]).repeat([num_batch, 1]).cuda()
                id = i * (grid_w + 1) + grid_w
                tar = torch.cat([theta[:, i:i+1, j:j+1, :],
                                theta[:, i:i+1, j+1:j+2, :],
                                theta[:, i+1:i+2, j:j+1, :],
                                theta[:, i+1:i+2, j+1:j+2, :]], dim=1)
                tar = tar.view([num_batch, 8])
                Hs.append(get_H(ori, tar).view([num_batch, 1, 9]))

        Hs = torch.cat(Hs, dim=1).view([num_batch, grid_h, grid_w, 9])
        return Hs

    def _meshgrid2(height, width, sh, eh, sw, ew):
        hn = eh - sh + 1
        wn = ew - sw + 1

        x_t = torch.matmul(torch.ones([hn, 1]).cuda(),
                           torch.transpose(torch.linspace(-1.0, 1.0, width)[sw:sw+wn].unsqueeze(1), 1, 0).cuda())
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height)[sh:sh+hn].unsqueeze(1).cuda(),
                           torch.ones([1, wn]).cuda())
        
        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def _transform3(theta, input_dim):
        input_dim = input_dim.permute([0, 2, 3, 1])
        num_batch = input_dim.size()[0]
        num_channels = input_dim.size()[3]
        theta = torch._cast_Float(theta)
        Hs = get_Hs(theta)
        gh = int(math.floor(height / grid_h))
        gw = int(math.floor(width / grid_w))
        x_ = []
        y_ = []

        for i in range(grid_h):
            row_x_ = []
            row_y_ = []
            for j in range(grid_w):
                H = Hs[:, i:i+1, j:j+1, :].view(num_batch, 3, 3)
                sh = i * gh
                eh = (i + 1) * gh - 1
                sw = j * gw
                ew = (j + 1) * gw - 1
                if (i == grid_h - 1):
                    eh = height - 1
                if (j == grid_w - 1):
                    ew = width - 1
                grid = _meshgrid2(height, width, sh, eh, sw, ew)
                grid = grid.unsqueeze(0)
                grid = grid.repeat([num_batch, 1, 1])

                T_g = torch.matmul(H, grid)
                x_s = T_g[:, 0:1, :]
                y_s = T_g[:, 1:2, :]
                z_s = T_g[:, 2:3, :]

                z_s_flat = z_s.contiguous().view(-1)
                t_1 = torch.ones(z_s_flat.size()).cuda()
                t_0 = torch.zeros(z_s_flat.size()).cuda()

                sign_z_flat = torch.where(z_s_flat >= 0, t_1, t_0) * 2 - 1
                z_s_flat = z_s.contiguous().view(-1) + sign_z_flat * 1e-8
                x_s_flat = x_s.contiguous().view(-1) / z_s_flat
                y_s_flat = y_s.contiguous().view(-1) / z_s_flat

                x_s = x_s_flat.view([num_batch, eh - sh + 1, ew - sw + 1])
                y_s = y_s_flat.view([num_batch, eh - sh + 1, ew - sw + 1])
                row_x_.append(x_s)
                row_y_.append(y_s)
            row_x = torch.cat(row_x_, dim=2)
            row_y = torch.cat(row_y_, dim=2)
            x_.append(row_x)
            y_.append(row_y)

        x = torch.cat(x_, dim=1).view([num_batch, height, width, 1])
        y = torch.cat(y_, dim=1).view([num_batch, height, width, 1])
        x_map_ = x.clone()
        y_map_ = y.clone()
        img = torch.cat([x, y], dim=3)
        x_s_flat = x.view(-1)
        y_s_flat = y.view(-1)

        t_1 = torch.ones(x_s_flat.size()).cuda()
        t_0 = torch.zeros(x_s_flat.size()).cuda()

        cond = (torch.gt(t_1 * -1, x_s_flat) | torch.gt(x_s_flat, t_1)) | \
            (torch.gt(t_1 * -1, y_s_flat) | torch.gt(y_s_flat, t_1))
        
        black_pix = torch.where(cond, t_1, t_0).view([num_batch, height, width])

        out_size = (height, width)
        input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

        output = input_transformed.view([num_batch, height, width, num_channels])
        output = output.permute([0, 3, 1, 2])

        return output, black_pix, img, x_map_, y_map_
    
    output = _transform3(theta, U)
    return output

class stabNet(nn.Module):
    def __init__(self):
        super(stabNet, self).__init__()
        self.resnet50 = KitModel()
        self.resnet50.resnet_v2_50_conv1_Conv2D = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
        self.regressor = nn.Sequential(nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, ((grid_h + 1) * (grid_w + 1) * 2)),
                                        )
        self.use_black_loss = 0
        self.one_ = torch.ones([batch_size, grid_h, grid_w, 8]).cuda()
        self.zero_ = torch.zeros([batch_size, grid_h, grid_w, 8]).cuda()

    def get_black_pos(self, pts):
        one_ = self.one_ / do_crop_rate
        zero_ = self.zero_
        black_err = torch.where(torch.gt(pts, one_), pts - one_, zero_) + \
                    torch.where(torch.gt(-1 * one_, pts), one_ * -1 - pts, zero_)
        return black_err.view([batch_size, -1])

    def get_4_pts(self, theta, batch_size):
        pts1_ = []
        pts2_ = []
        pts = []
        h = 2.0 / grid_h
        w = 2.0 / grid_w
        tot = 0
        for i in range(grid_h + 1):
            pts.append([])
            for j in range(grid_w + 1):
                hh = i * h - 1
                ww = j * w - 1
                p = torch._cast_Float(torch.Tensor([ww, hh]).view(2)).cuda()
                temp = theta[:, tot * 2: tot * 2 + 2]
                tot += 1
                p = (p + temp).view([batch_size, 1, 2])
                p = torch.clamp(p, -1. / do_crop_rate, 1. / do_crop_rate)
                pts[i].append(p.view([batch_size, 2, 1]))
                pts2_.append(p)

        for i in range(grid_h):
            for j in range(grid_w):
                g = torch.cat([pts[i][j], pts[i][j + 1], pts[i + 1][j], pts[i + 1][j + 1]], dim = 2)
                pts1_.append(g.view([batch_size, 1, 8]))
        
        pts1 = torch.cat(pts1_, 1).view([batch_size, grid_h, grid_w, 8])
        pts2 = torch.cat(pts2_, 1).view([batch_size, grid_h + 1, grid_w + 1, 2])

        return pts1, pts2

    def warp_pts(self, pts, flow):
        x = pts[:, :, 0]
        x = torch.clamp((x + 1) / 2 * width, 0, width - 1)
        x = torch._cast_Int(torch.round(x))
        y = pts[:, :, 1]
        y = torch.clamp((y + 1) / 2 * height, 0, height - 1)
        y = torch._cast_Int(torch.round(y))

        out = []
        for i in range(batch_size):
            flow_ = flow[i, :, :, :].view([-1, 2])
            xy = x[i, :] + y[i, :] * width
            xy = torch._cast_Long(xy)
            temp = flow_[xy]
            out.append(temp.view([1, max_matches, 2]))
        return torch.cat(out, 0)

    def forward(self, x_tensor):
        x_batch_size = x_tensor.size()[0]
        x = x_tensor[:, 12:13, :, :]

        # summary 1, dismiss now
        x_tensor = self.resnet50(x_tensor)
        x_tensor = torch.mean(x_tensor, dim=[2, 3])
        theta = self.regressor(x_tensor)

        pts1, pts2 = self.get_4_pts(theta, x_batch_size)

        h_trans, black_pix, flow, x_map, y_map = transformer(x, pts2) # NCHW NHWC NHWC

        return h_trans.permute([0, 2, 3, 1]), black_pix, x_map, y_map
