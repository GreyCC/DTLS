import os
import glob
import cv2
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

input_folder = "../mse_test_result"
ground_truth_folder = "16_128/16_128_exp"
psnr_avg = 0
ssim_avg = 0
total = 0
# lpips_alex = lpips.LPIPS(net='alex')
l2_avg = 0
transform = transforms.ToTensor()

for idx, path in enumerate(sorted(glob.glob(os.path.join(input_folder, '*')))):
    total += 1
    imgname = os.path.splitext(os.path.basename(path))[0]
    tmp = imgname.split('_')[-1]
    gt_imgname = f"original_{tmp}"

    print('Evaluating', idx, imgname)
    rs = cv2.imread(f'{input_folder}/{imgname}.png')
    gt = cv2.imread(f'{ground_truth_folder}/{gt_imgname}.png')

    psnr = compare_psnr(gt, rs, data_range=255)
    ssim = compare_ssim(gt, rs, channel_axis=-1)

    rs = cv2.resize(rs,(16,16),interpolation=cv2.INTER_CUBIC)
    gt = cv2.resize(gt,(16,16),interpolation=cv2.INTER_CUBIC)

    l2 = (gt - rs) ** 2
    l2 = np.sum(l2)
    l2_avg += l2

    psnr_avg += psnr
    ssim_avg += ssim

print("SSIM: ", ssim_avg/total, " | PSNR: ", psnr_avg/total, " | Consistency:", l2_avg/total)
