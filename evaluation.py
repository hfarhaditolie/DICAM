
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio,mean_squared_error

import numpy as np 
from libsvm.svmutil import *
import matplotlib.pyplot as plt
import cv2
import os
import skimage
import imageio as iio
from uiqm import *
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


generated = []
gt= []

gt_addrr = "directory path of ground-truth images"
addrr = "directory path of generated images"

for item in os.listdir(addrr):
	if item.endswith(".png"):
		generated.append((((cv2.cvtColor(cv2.imread(addrr+item), cv2.COLOR_BGR2RGB).astype("float32")))))
for item in os.listdir(gt_addrr):
	if ".png" in item:
		gt.append((cv2.cvtColor(cv2.imread(gt_addrr+item), cv2.COLOR_BGR2RGB).astype("float32")))

SSIM_results = []
PSNR_results = []
UIQM = []
UCIQE= []
MSE = []
for i in range(len(generated)):

	print(i)
	UIQM.append(getUIQM(NormalizeData(generated[i])))
	UCIQE.append(getUCIQE(NormalizeData(generated[i])))

	PSNR = peak_signal_noise_ratio(NormalizeData(generated[i]),NormalizeData(gt[i]))
	PSNR_results.append(PSNR)

	SSIM = structural_similarity(NormalizeData(generated[i]),NormalizeData(gt[i]),multichannel=True)
	SSIM_results.append(SSIM)

	MSE.append(mean_squared_error(NormalizeData(generated[i]),NormalizeData(gt[i])))
	
print(np.mean(SSIM_results), np.mean(PSNR_results), np.mean(MSE),np.mean(UIQM),np.mean(UCIQE))


