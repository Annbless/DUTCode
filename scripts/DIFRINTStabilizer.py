import argparse
import os
import sys
from shutil import copyfile

import torch
import torch.nn as nn
from torch.autograd import Variable
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
from models.DIFRINT.models import DIFNet2
from models.DIFRINT.pwcNet import PwcNet

from PIL import Image
import numpy as np
import math
import pdb
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--modelPath', default='./trained_models/DIFNet2.pth')  # 2
parser.add_argument('--InputBasePath', default='')
parser.add_argument('--OutputBasePath', default='./')
parser.add_argument('--temp_file', default='./DIFRINT_TEMP/')
parser.add_argument('--n_iter', type=int, default=3,
                    help='number of stabilization interations')
parser.add_argument('--skip', type=int, default=2,
                    help='number of frame skips for interpolation')
parser.add_argument('--desiredWidth', type=int, default=640,
                    help='width of the input video')
parser.add_argument('--desiredHeight', type=int, default=480,
                    help='height of the input video')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##########################################################

# Networks
DIFNet = DIFNet2()

# Place Network in cuda memory
if opt.cuda:
    DIFNet.cuda()

# DataParallel
DIFNet = nn.DataParallel(DIFNet)
DIFNet.load_state_dict(torch.load(opt.modelPath))
DIFNet.eval()

if not os.path.exists(opt.OutputBasePath):
    os.mkdir(opt.OutputBasePath)

if not os.path.exists(opt.temp_file):
    os.mkdir(opt.temp_file)

##########################################################

frameList = [ele for ele in os.listdir(opt.InputBasePath) if ele[-4:] == '.jpg']
frameList = sorted(frameList, key=lambda x: int(x[:-4]))

if os.path.exists(opt.temp_file):
    copyfile(opt.InputBasePath + frameList[0], opt.temp_file + frameList[0])
    copyfile(opt.InputBasePath + frameList[-1], opt.temp_file + frameList[-1])
else:
    os.makedirs(opt.temp_file)
    copyfile(opt.InputBasePath + frameList[0], opt.temp_file + frameList[0])
    copyfile(opt.InputBasePath + frameList[-1], opt.temp_file + frameList[-1])
# end

# Generate output sequence
for num_iter in range(opt.n_iter):
    idx = 1
    print('\nIter: ' + str(num_iter+1))
    for f in frameList[1:-1]:
        if f.endswith('.jpg'):
            if num_iter == 0:
                src = opt.InputBasePath
            else:
                src = opt.temp_file
            # end

            if idx < opt.skip or idx > (len(frameList)-1-opt.skip):
                skip = 1
            else:
                skip = opt.skip


            fr_g1 = torch.cuda.FloatTensor(np.array(Image.open(opt.temp_file + '%d.jpg' % (
                int(f[:-4])-skip)).resize((opt.desiredWidth, opt.desiredHeight))).transpose(2, 0, 1).astype(np.float32)[None, :, :, :] / 255.0)

            fr_g3 = torch.cuda.FloatTensor(np.array(Image.open(
                src + '%d.jpg' % (int(f[:-4])+skip)).resize((opt.desiredWidth, opt.desiredHeight))).transpose(2, 0, 1).astype(np.float32)[None, :, :, :] / 255.0)


            fr_o2 = torch.cuda.FloatTensor(np.array(Image.open(
                opt.InputBasePath + f).resize((opt.desiredWidth, opt.desiredHeight))).transpose(2, 0, 1).astype(np.float32)[None, :, :, :] / 255.0)

            with torch.no_grad():
                fhat, I_int = DIFNet(fr_g1, fr_g3, fr_o2,
                                     fr_g3, fr_g1, 0.5)  # Notice 0.5

            # Save image
            img = Image.fromarray(
                np.uint8(fhat.cpu().squeeze().permute(1, 2, 0)*255))
            img.save(opt.temp_file + f)

            sys.stdout.write('\rFrame: ' + str(idx) +
                             '/' + str(len(frameList)-2))
            sys.stdout.flush()

            idx += 1
        # end
    # end

frame_rate = 25
frame_width = opt.desiredWidth
frame_height = opt.desiredHeight

print("generate stabilized video...")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(opt.OutputBasePath + '/DIFRINT_stable.mp4', fourcc, frame_rate, (frame_width, frame_height))

for f in frameList:
    if f.endswith('.jpg'):
        img = cv2.imread(os.path.join(opt.temp_file, f))
        out.write(img)

out.release()
