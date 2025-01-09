import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from lib.pranet import PraNet_V2, PVT_PraNet_V2
from MyTest_med import test_with_eval

torch.backends.cudnn.enabled = False


def structure_loss(pred, pred_bg, mask_fg, mask_bg):
    # Weight (average pooling to get a smoothed version, calculate the absolute difference with the original mask, multiply by 5 to amplify the penalty for boundary differences, and finally add 1 to ensure the minimum weight is 1)
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask_fg, kernel_size=31, stride=1, padding=15) - mask_fg) # Add 1 to ensure the minimum weight is 1

    # Weighted binary cross-entropy loss (foreground)
    wbce = F.binary_cross_entropy_with_logits(pred, mask_fg, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) # 归一化
    
    # Weighted binary cross-entropy loss (background)
    wbce2 = F.binary_cross_entropy_with_logits(pred_bg, mask_bg, reduction='none')
    wbce2 = (weit*wbce2).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) # 归一化

    pred = torch.sigmoid(pred)
    
    # Weighted intersection over union loss
    inter = ((pred * mask_fg)*weit).sum(dim=(2, 3))
    union = ((pred + mask_fg)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou + 0.8*wbce2).mean()

def show_tensor(tensor_list,path=None):
    for i, tensor in enumerate(tensor_list):
        if isinstance(tensor,np.ndarray):
            array=tensor
        else:
            array=np.array(tensor.to('cpu'))
        if path is None:
            np.savetxt('tensor_values'+str(i+1)+'.txt', array, fmt='%0.6f')
        else:
            np.savetxt(path, array, fmt='%0.6f')


def train(train_loader, model, optimizer, epoch, opt):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts)
            
            # bg_mask=1-gts
            # show_tensor([bg_mask[0].squeeze(0)], 'bg_mask.txt')
            
            gts = gts.cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            bg_mask = 1-gts
            # ---- forward ----
            lateral_map_2_fg,lateral_map_3_fg,lateral_map_4_fg,lateral_map_5_fg,lateral_map_2_bg,lateral_map_3_bg,lateral_map_4_bg,lateral_map_5_bg = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_2_fg, lateral_map_2_bg, gts, bg_mask)
            loss4 = structure_loss(lateral_map_3_fg, lateral_map_3_bg, gts, bg_mask)
            loss3 = structure_loss(lateral_map_4_fg, lateral_map_4_bg, gts, bg_mask)
            loss2 = structure_loss(lateral_map_5_fg, lateral_map_5_bg, gts, bg_mask)
            loss = loss2 + loss3 + loss4 + loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PraNetV2-%d.pth' % (epoch))
        print('[Saving Snapshot:]', save_path + 'PraNetV2-%d.pth'% (epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=30, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='PraNetV2_res') # TODO: Change the name of the folder to save the model
    parser.add_argument('--model_type', type=str, default='PraNet-V2') # TODO: Choose which model to train【PraNet-V2 or PVT-PraNet-V2】
    opt = parser.parse_args()

    # ---- load data ----
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    print("length of train dataset: {}".format(len(train_loader)))
    total_step = len(train_loader)
    print("#"*20, "Start Training", "#"*20)
    import sys
    sys.exit(0)
    
    # ---- build models ----
    if opt.model_type == 'PraNet-V2':
        model = PraNet_V2(num_class=1).cuda()
    elif opt.model_type == 'PVT-PraNet-V2':
        model = PVT_PraNet_V2(num_class=1).cuda()
    else:
        raise ValueError('Model Not Found, choose from [PraNet-V2, PVT-PraNet-V2]')


    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)


    eval_config={
    "datasets": ['CVC-300', 'CVC-ClinicDB'],
    "metrics": ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae'],
    "test_size":352,
    }
  
    best_eval_res = [[0, 0, 0], [0, 0, 0]]
    for epoch in tqdm(range(1, opt.epoch), desc="Training Epochs"):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt)
        
        print('Evaluating the model...')
        eval_res = test_with_eval(eval_config, model)
        print('Epoch: %d, Evaluation:\n %s' % (epoch, eval_res))
        
        if eval_res[0][0] + eval_res[1][0] - best_eval_res[0][0] - best_eval_res[1][0] > 0:
        # if eval_res[0][0] - best_eval_res[0][0]  > 0:
            best_eval_res = eval_res
            save_path = 'snapshots/{}/'.format(opt.train_save)
            torch.save(model.state_dict(), save_path + 'best.pth')
            print('[Saving Snapshot:]', save_path + 'best.pth in epoch %d' % (epoch + 1))
        
        

            

