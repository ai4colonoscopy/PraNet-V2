import os, argparse, sys
import imageio
import numpy as np
import time


import jittor as jt
from jittor import nn


from lib.PraNet_Res2Net import PraNet, PVT_PraNet
from lib.pranet import PraNet_V2,PVT_PraNet_V2
from utils.dataloader import test_dataset
from jittor.dataset import DataLoader

from tqdm import tqdm
import imageio
import cv2 
import numpy as np
from PIL import Image

jt.flags.use_cuda = 1


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--bs', type=int, default=1, help='testing size')
opt = parser.parse_args()

# ---- build models ----
model_names = ['PraNet-V1', 'PVT-PraNet-V1', 'PraNet-V2', 'PVT-PraNet-V2']
pth_path_list = ['../snapshots/PraNet-V1/RES-V1.pth', '../snapshots/PVT-PraNet-V1/PVT-V1.pth', '../snapshots/PraNet-V2/RES-V2.pth', '../snapshots/PVT-PraNet-V2/PVT-V2.pth']
model1 = PraNet()
model1.load_state_dict(jt.load(pth_path_list[0]))
model1.cuda()
model1.eval()

model2 = PVT_PraNet()
model2.load_state_dict(jt.load(pth_path_list[1]))
model2.cuda()
model2.eval()

model3 = PraNet_V2(num_class=1)
saved_model = jt.load(pth_path_list[2])
model_dict = model3.state_dict()
state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model3.load_state_dict(model_dict)
model3.cuda()
model3.eval()

model4 = PVT_PraNet_V2(num_class=1)
saved_model = jt.load(pth_path_list[3])
model_dict = model4.state_dict()
state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model4.load_state_dict(model_dict)
model4.cuda()
model4.eval()
model_list=[model1,model2,model3,model4]
print("Model Loaded Successfully")

total_time = 0
total_images = 0
perdataset_time = 0
perdataset_images = 0

dataset_list = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'ETIS-LaribPolypDB']
if opt.bs > 1: dataset_list = ['CVC-300', 'CVC-ClinicDB', 'ETIS-LaribPolypDB']
for model_name, model in zip(model_names, model_list):
    total_time = 0
    total_images = 0
    for _data_name in tqdm(dataset_list, desc=f"[{model_name}:]Testing in Datasets"):
        data_path = '../data/TestDataset/{}/'.format(_data_name)
        save_path = './results/{}/{}/'.format(model_name,_data_name)
        os.makedirs(save_path, exist_ok=True)
        
        perdataset_images = 0
        perdataset_time = 0
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        dataset = test_dataset(image_root, gt_root, opt.testsize) 
        test_loader = DataLoader(dataset, batch_size=opt.bs, shuffle=False)

        for image, gt, name in test_loader:
            # print(image.shape, gt.shape)
            gt /= (gt.max() + 1e-08)
            
            
            if model_name in ['PraNet-V1', 'PVT-PraNet-V1']:
                start_time = time.time()
                (res5, res4, res3, res2) = model(image)
                end_time = time.time() 
                res = res2
            else:
                start_time = time.time()
                res2, res3, res4, res5, res2_bg, res3_bg, res4_bg, res5_bg = model(image)
                end_time = time.time() 
                res = res2 + res3 + res4 + res5
            perdataset_time += (end_time - start_time)
            perdataset_images += opt.bs
            
            c, h, w = gt.shape
            res = nn.upsample(res, size=(h, w), mode='bilinear')
            res = res.sigmoid().data.squeeze()
            res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
            res_uint8 = (res * 255).astype(np.uint8)
            if res_uint8.ndim == 3:
                for j in range(res_uint8.shape[0]):
                    imageio.imwrite((save_path + name[j]), res_uint8[j])
            else:
                imageio.imwrite((save_path + name[0]), res_uint8)
            
        # Calculate the FPS for each dataset.
        fps = perdataset_images / perdataset_time
        # print(f"{model_name} - {_data_name} - FPS: {fps:.2f}")
        total_time += perdataset_time
        total_images += perdataset_images
    
    # Calculate the FPS of the model across all datasets.
    fps = total_images / total_time
    print(f"{model_name} - FPS: {fps:.2f}")
    