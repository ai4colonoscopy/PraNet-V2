import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset

import imageio
from tqdm import tqdm

from lib.pranet import PraNet_V2,PVT_PraNet_V2
from eval import eval_for_testAllInOne

def test_with_eval(eval_config,model):
    
    res = np.zeros((len(eval_config['datasets']),len(eval_config['metrics'])))
    datasets_list = eval_config['datasets']
    
    for i,_data_name in tqdm(enumerate(datasets_list)):
        tqdm.write(f"Testing in Datasets {_data_name}:")
        data_path = './data/TestDataset/{}/'.format(_data_name)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, eval_config["test_size"])
        res_i = np.zeros((len(test_loader),len(eval_config['metrics'])))
        
        model.eval()
        # for j in range(test_loader.size):
        for j in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            p2, p3, p4, p5, _, _, _, _ = model(image)
            output = p2+p3+p4+p5
            output = F.interpolate(output, size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)


            output_uint8 = (output * 255).astype(np.uint8)
            
            res_i[j,:] = eval_for_testAllInOne(eval_config,output_uint8, gt)
        res[i] = np.mean(res_i, axis=0)
    return res
            
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    # parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet-V2/RES-V2.pth')
    # parser.add_argument('--pth_path', type=str, default='./snapshots/PVT-PraNet-V2/PVT-V2.pth')
    parser.add_argument('--pth_path', type=str, default='./snapshots/PVT-PraNet-V2/PVT-V2-ep20.pth') # TODO: Replace with the path to the model you want to test
    

    for _data_name in tqdm(['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'ETIS-LaribPolypDB'],desc="Testing in Datasets:"):

        data_path = './data/TestDataset/{}/'.format(_data_name)
        save_path = './results/PVT-PraNet-V2-ep20/{}/'.format(_data_name) # TODO: Change the name of the folder to save the segmentation results
        opt = parser.parse_args()
        # model = PraNet()
        model = PVT_PraNet_V2(num_class=1)
        model.load_state_dict(torch.load(opt.pth_path))
        model.cuda()
        model.eval()

        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res2, res3, res4, res5, res2_bg, res3_bg, res4_bg, res5_bg = model(image)
            res = res2 + res3 + res4 + res5
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)


            res_uint8 = (res * 255).astype(np.uint8)
        
            imageio.imwrite(save_path + name, res_uint8)