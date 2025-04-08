import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc
import time
import imageio
from model.SFINet_V3 import SFINet
from utils.data import test_dataset

parser = argparse.ArgumentParser()
argument = parser.add_argument('--testsize', type=int, default=352, help='testing size')
args = parser.parse_args()
opt = args

dataset_path = 'E:/Datasets/'

model = SFINet()
model.load_state_dict(torch.load('./models/SFINet_V3_ORSI-4199.pth', map_location=torch.device('cuda:0')))

model.cuda()
model.eval()

test_datasets = ['ORSI-4199']   # 'ECSSD', 'PASCALS', 'DUT-O', 'HKU-IS', 'DUTS-TE', 'ORSI-4199', 'EORSSD', 'ORSSD'

for dataset in test_datasets:
    save_path = './results/' + 'SFINet_V3_' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test-images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        x_out, x_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.interpolate(x_out[2], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))
