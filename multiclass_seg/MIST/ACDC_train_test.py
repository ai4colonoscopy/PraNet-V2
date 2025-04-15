#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time
import torch, gc
import numpy as np
from tqdm import tqdm
from medpy.metric import dc,hd95
from scipy.ndimage import zoom
from datetime import datetime
import pytz

from utils.utils import powerset
from utils.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
# from test_ACDC import inference
from lib.networks import MIST_CAM


def convert_labels_to_one_hot_masks(batch_labels, num_classes):
    bs, H, W = batch_labels.shape
    one_hot_masks = torch.zeros((bs, num_classes, H, W), dtype=torch.uint8, device=batch_labels.device)

    # Perform one-hot encoding using the scatter method: set corresponding positions in one_hot_masks to 1 based on pixel values in batch_labels as indices.
    one_hot_masks.scatter_(1, batch_labels.unsqueeze(1), 1)
    inverted_masks = torch.logical_not(one_hot_masks).to(torch.uint8)
    
    return inverted_masks


def val():
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]

        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(0).cpu().detach().numpy()

        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3) # not for double_maxvits
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        
        if args.dual:
            P = net(val_image_batch)[:4]
            P_bg=net(val_image_batch)[-4:]
            #print(len(P))
        
            val_outputs = 0.0
            for idx in range(len(P)):
                val_outputs += (P[idx]-P_bg[idx])
        else:
            P = net(val_image_batch)

            val_outputs = 0.0
            for idx in range(len(P)):
                val_outputs += P[idx]
        
        val_outputs = torch.softmax(val_outputs, dim=1)

        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs

        dc_sum+=dc(val_outputs,val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f' % (performance))

    print('Testing performance in val model: mean_dice : %f' % (performance))
    #print("val avg_dsc: %f" % (performance))
    return performance


from utils.utils import test_single_volume
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,use_dual=args.dual)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
            # print('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
            # logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
            print('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        
        logging.info("Testing Finished!")
        print("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd




# In[2]:


gc.collect()
torch.cuda.empty_cache()
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'


# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                    'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                    'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)
parser.add_argument("--batch_size", default=12, help="batch size") #12
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=400)
parser.add_argument("--img_size", default=256)
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="/path/to/lists_ACDC") # TODO: replace with actual path
parser.add_argument("--root_dir", default="/path/to/ACDC/") # TODO: replace with actual path (root of ACDC dataset)
parser.add_argument("--volume_path", default="/path/to/ACDC/test") # TODO: replace with actual path
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--is_savefig', default=False, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
# parser.add_argument('--test_save_dir', default=None, help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')      
parser.add_argument('--dual', default=True, action='store_true', help='dual supervision or single supervision')      
args = parser.parse_args()


# In[4]:

if(args.test_save_dir is None):
    print("##########Disable test_save##########")
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



args.is_pretrain = True
args.exp = 'MIST_CAM_loss_MUTATION_w3_7_' + str(args.img_size)
if args.dual:
    args.exp = 'Dual_' + args.exp

current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H")
print("The current time is", current_time)
args.exp='run:'+current_time+'_'+args.exp

snapshot_path = "{}/{}".format(args.save_path, args.exp)
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_'+str(args.img_size)
snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
    
    
logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

if args.is_savefig:
    test_save_path = os.path.join(snapshot_path, 'predictions')
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path, exist_ok=True)
else:
    logging.info("##########Disable savefig##########")
    test_save_path = None

if args.dual:
    logging.info("########## Using Dual Supervision ##########")
    net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear',dual=True).cuda()
else:
    logging.info("########## Using Single Supervision##########")
    net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()


# In[5]:



if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)


# In[6]:


net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
bce_loss = nn.BCEWithLogitsLoss()
#save_interval = args.n_skip

# iterator = tqdm(range(0, args.max_epochs), ncols=70)
# iter_num = 0

Loss = []
Test_Accuracy = []

# Best_dcs = 0.80
# Best_dcs_th = 0.865
Best_dcs=0.87
Best_interface=0.91


max_iterations = args.max_epochs * len(train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
#optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)


# In[7]:

l = [0, 1, 2, 3]
ss = [x for x in powerset(l)] # for mutation
#ss = [[0],[1],[2],[3]] # for only four-stage loss, no mutation
#print(ss)
iter_num = 0
best_state_dict=net.state_dict()
for epoch in tqdm(range(args.max_epochs)):
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        bg_mask=convert_labels_to_one_hot_masks(label_batch, args.num_classes)
        
        image_batch, label_batch, bg_mask = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor), bg_mask.type(torch.FloatTensor)
        image_batch, label_batch, bg_mask = image_batch.cuda(), label_batch.cuda(), bg_mask.cuda()
        
        if args.dual:
            P = net(image_batch)
            P_fg = P[:4]   # 提取前四个，即 p1, p2, p3, p4
            P_bg = P[-4:]   # 提取后四个，即 p1_bg, p2_bg, p3_bg, p4_bg
            loss = 0.0
            lc1, lc2, lc3 = 0.5, 0.7, 0.3
            # lc1, lc2, lc3 = 0.6, 0.7, 0.4
            # lc1, lc2, lc3 = 1.0，1.0，1.0
        
            for s in ss:
                iout = 0.0
                ibg=0.0
                if(s==[]):
                    continue
                #print(s)
                for idx in range(len(s)):
                    iout += P_fg[s[idx]]
                    ibg += P_bg[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                # loss_dice = dice_loss(iout, label_batch, softmax=True)                
                loss_bce=bce_loss(ibg, bg_mask[:])
                
                # if not is_same_order_of_magnitude(loss_ce.item(), loss_dice.item(), loss_bce.item()):
                #     print("=====Warning 3 loss not in same magnitude ce:{}/dice:{}/bce:{}=====".format(loss_ce,loss_dice,loss_bce))
                #     logging.info("=====Warning 3 loss not in same magnitude ce:{}/dice:{}/bce:{}=====".format(loss_ce,loss_dice,loss_bce))

                loss += (lc1 * loss_ce + lc2 * loss_dice + lc3 * loss_bce) # 原始
                # loss += (0.7 * loss_dice + 0.3 * loss_bce) # bce+dice
                # loss += (0.7 * loss_dice + 0.3 * loss_ce) # ce+dice
                # loss += (0.7 * loss_ce + 0.3 * loss_bce) # ce+bce
                # loss += (lc1 * loss_ce + lc3 * loss_bce) 
        else:
            P = net(image_batch)
            loss = 0.0
            lc1, lc2 = 0.3, 0.7
                    
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                #print(s)
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (lc1 * loss_ce + lc2 * loss_dice) 
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # We did not use this
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num%50 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss/len(train_dataset))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    
    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)

    
    if  epoch > 0.25 * args.max_epochs:
        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, test_save_path)
        if avg_test_dcs > Best_interface:
            Best_interface = avg_test_dcs
            save_model_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            print("save model to {}".format(save_model_path))
        print("test avg_dsc: %f, best avg_dsc:%f" % (avg_test_dcs, Best_interface))
        logging.info("test avg_dsc: %f, best avg_dsc:%f" % (avg_test_dcs, Best_interface))
        Test_Accuracy.append(avg_test_dcs)  



    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path,  'epoch={}_lr={}.pth'.format(epoch, lr_))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        break
    
# Write loss to the log
logging.info("Loss trend:")
for loss in Loss:
    logging.info(f"Loss: {loss}")