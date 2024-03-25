from Config.models import DICAM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from Config.options import opt
import math
import shutil
from tqdm import tqdm

CHECKPOINTS_DIR = opt.checkpoints_dir
INP_DIR = opt.testing_dir_inp
CLEAN_DIR = opt.testing_dir_gt

device = 'cuda' if torch.cuda.is_available() else 'cpu'        

ch = 3

network = DICAM()
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"DICAM_60.pt"),map_location=torch.device('cpu'))
network.load_state_dict(checkpoint['model_state_dict'])
network.eval()
network.to(device)

result_dir = 'results/UIEB_DICAM/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if __name__ =='__main__':

    total_files = os.listdir(INP_DIR)
    st = time.time()
    with tqdm(total=len(total_files)) as t:

        for m in total_files:
        
            img=cv2.imread(INP_DIR + str(m))
            img = cv2.resize(img, (256,256),
                                 interpolation=cv2.INTER_AREA)
            img = img[:, :, ::-1]   
            img = np.float32(img) / 255.0
            h,w,c=img.shape

            train_x = np.zeros((1, ch, h, w)).astype(np.float32)

            train_x[0,0,:,:] = img[:,:,0]
            train_x[0,1,:,:] = img[:,:,1]
            train_x[0,2,:,:] = img[:,:,2]
            dataset_torchx = torch.from_numpy(train_x)
            dataset_torchx=dataset_torchx.to(device)

            output=network(dataset_torchx)
            output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = output[:, :, ::-1]
            cv2.imwrite(os.path.join(result_dir + str(m)), output)

            t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
            t.update(1)
            
    end = time.time()
    print('Total time taken in secs : '+str(end-st))
    print('Per image (avg): '+ str(float((end-st)/len(total_files))))
