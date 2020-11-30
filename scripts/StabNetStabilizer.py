import torch
import torch.nn as nn
import argparse
from PIL import Image
import cv2
import os
import traceback
import math
import time
import sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from models.StabNet.v2_93 import *
from models.StabNet.model import stabNet

parser = argparse.ArgumentParser()
parser.add_argument('--modelPath', default='./models')
parser.add_argument('--before-ch', type=int)
parser.add_argument('--OutputBasePath', default='data_video_local')
parser.add_argument('--InputBasePath', default='')
parser.add_argument('--max-span', type=int, default=1)
parser.add_argument('--refine', type=int, default=1)
parser.add_argument('--no_bm', type=int, default=1)
args = parser.parse_args()

MaxSpan = args.max_span
args.indices = indices[1:]
batch_size = 1

before_ch = max(args.indices)#args.before_ch
after_ch = max(1, -min(args.indices) + 1)

model = stabNet()
r_model = torch.load(args.modelPath)
model.load_state_dict(r_model)
model.cuda()
model.eval()

def cvt_img2train(img, crop_rate = 1):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    if (crop_rate != 1):
        h = int(height / crop_rate)
        dh = int((h - height) / 2)
        w = int(width / crop_rate)
        dw = int((w - width) / 2)

        img = img.resize((w, h), Image.BILINEAR)
        img = img.crop((dw, dh, dw + width, dh + height))
    else:
        img = img.resize((width, height), Image.BILINEAR)
    img = np.array(img)
    img = img * (1. / 255) - 0.5
    img = img.reshape((1, height, width, 1))
    return img

def make_dirs(path):
    if not os.path.exists(path): os.makedirs(path)

cvt_train2img = lambda x: ((np.reshape(x, (height, width)) + 0.5) * 255).astype(np.uint8)

def warpRevBundle2(img, x_map, y_map):
    assert(img.ndim == 3)
    assert(img.shape[-1] == 3)
    rate = 4
    x_map = cv2.resize(cv2.resize(x_map, (int(width / rate), int(height / rate))), (width, height))
    y_map = cv2.resize(cv2.resize(y_map, (int(width / rate), int(height / rate))), (width, height))
    x_map = (x_map + 1) / 2 * width
    y_map = (y_map + 1) / 2 * height
    dst = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR)
    assert(dst.shape == (height, width, 3))
    return dst

production_dir = args.OutputBasePath
make_dirs(production_dir)

image_len = len([ele for ele in os.listdir(args.InputBasePath) if ele[-4:] == '.jpg'])
images = []

for i in range(image_len):

    image = cv2.imread(os.path.join(args.InputBasePath, '{}.jpg'.format(i)))
    image = cv2.resize(image, (width, height))
    images.append(image)

print('inference with {}'.format(args.indices))

tot_time = 0

print('totally {} frames for stabilization'.format(len(images)))

before_frames = []
before_masks = []
after_frames = []
after_temp = []

cnt = 0

frame = images[cnt]

cnt += 1

for i in range(before_ch):
    before_frames.append(cvt_img2train(frame, crop_rate))
    before_masks.append(np.zeros([1, height, width, 1], dtype=np.float))
    temp = before_frames[i]
    temp = ((np.reshape(temp, (height, width)) + 0.5) * 255).astype(np.uint8)

    temp = np.concatenate([temp, np.zeros_like(temp)], axis=1)
    temp = np.concatenate([temp, np.zeros_like(temp)], axis=0)


for i in range(after_ch):
    frame = images[cnt]
    cnt = cnt + 1
    frame_unstable = frame
    after_temp.append(frame)
    after_frames.append(cvt_img2train(frame, 1))

length = 0
in_xs = []
delta = 0

dh = int(height * 0.8 / 2)
dw = int(width * 0.8 / 2)
all_black = np.zeros([height, width], dtype=np.int64)
frames = []

black_mask = np.zeros([dh, width], dtype=np.float)
temp_mask = np.concatenate([np.zeros([height - 2 * dh, dw], dtype=np.float), np.ones([height - 2 * dh, width - 2 * dw], dtype=np.float), np.zeros([height - 2 * dh, dw], dtype=np.float)], axis=1)
black_mask = np.reshape(np.concatenate([black_mask, temp_mask, black_mask], axis=0),[1, height, width, 1]) 

try:
    while(True):

        in_x = []
        if input_mask:
            for i in args.indices:
                if (i > 0):
                    in_x.append(before_masks[-i])
        for i in args.indices:
            if (i > 0):
                in_x.append(before_frames[-i])
        in_x.append(after_frames[0])
        for i in args.indices:
            if (i < 0):
                in_x.append(after_frames[-i])
        if (args.no_bm == 0):
            in_x.append(black_mask)
        # for i in range(after_ch + 1):
        in_x = np.concatenate(in_x, axis = 3)
        # for max span
        if MaxSpan != 1:
            in_xs.append(in_x)
            if len(in_xs) > MaxSpan: 
                in_xs = in_xs[-1:]
                print('cut')
            in_x = in_xs[0].copy()
            in_x[0, ..., before_ch] = after_frames[0][..., 0]
        tmp_in_x = np.array(in_x.copy())
        for j in range(args.refine):
            start = time.time()
            img, black, x_map_, y_map_ = model.forward(torch.Tensor(tmp_in_x.transpose((0, 3, 1, 2))).cuda())
            img = img.cpu().clone().detach().numpy()
            black = black.cpu().clone().detach().numpy()
            x_map_ = x_map_.cpu().clone().detach().numpy()
            y_map_ = y_map_.cpu().clone().detach().numpy()
            tot_time += time.time() - start
            black = black[0, :, :]
            xmap = x_map_[0, :, :, 0]
            ymap = y_map_[0, :, :, 0]
            all_black = all_black + np.round(black).astype(np.int64)
            img = img[0, :, :, :].reshape(height, width)
            frame = img + black * (-1)
            frame = frame.reshape(1, height, width, 1)
            tmp_in_x[..., -1] = frame[..., 0]
        img = ((np.reshape(img + 0.5, (height, width))) * 255).astype(np.uint8)
        
        net_output = img

        img_warped = warpRevBundle2(cv2.resize(after_temp[0], (width, height)), xmap, ymap)
        frames.append(img_warped)

        if cnt + 1 <= len(images):
            frame_unstable = images[cnt]
            cnt = cnt + 1
            ret = True
        else:
            ret = False  
        
        if (not ret):
            break
        length = length + 1
        if (length % 10 == 0):
            print("length: " + str(length))      
            print('fps={}'.format(length / tot_time))

        before_frames.append(frame)
        before_masks.append(black.reshape((1, height, width, 1)))
        before_frames.pop(0)
        before_masks.pop(0)
        after_frames.append(cvt_img2train(frame_unstable, 1))
        after_frames.pop(0)
        after_temp.append(frame_unstable)
        after_temp.pop(0)
except Exception as e:
    traceback.print_exc()
finally:
    print('total length={}'.format(length + 2))

    black_sum = np.zeros([height + 1, width + 1], dtype=np.int64)
    for i in range(height):
        for j in range(width):
            black_sum[i + 1][j + 1] = black_sum[i][j + 1] + black_sum[i + 1][j] - black_sum[i][j] + all_black[i][j]
    max_s = 0
    ans = []
    for i in range(0, int(math.floor(height * 0.5)), 10):
        print(i)
        print(max_s)
        for j in range(0, int(math.floor(width * 0.5)), 10):
            if (all_black[i][j] > 0):
                continue
            for hh in range(i, height):
                dw = int(math.floor(float(max_s) / (hh - i + 1)))
                for ww in range(j, width):
                    if (black_sum[hh + 1][ww + 1] - black_sum[hh + 1][j] - black_sum[i][ww + 1] + black_sum[i][j] > 0):
                        break
                    else:
                        s = (hh - i + 1) * (ww - j + 1)
                        if (s > max_s):
                            max_s = s
                            ans = [i, j, hh, ww]
    videoWriter = cv2.VideoWriter(os.path.join(production_dir, 'StabNet_stable.mp4'), 
        cv2.VideoWriter_fourcc(*'MP4V'), 25, (ans[3] - ans[1] + 1, ans[2] - ans[0] + 1))
    for frame in frames:
        frame_ = frame[ans[0]:ans[2] + 1, ans[1]:ans[3] + 1, :]
        videoWriter.write(frame_)
    videoWriter.release()