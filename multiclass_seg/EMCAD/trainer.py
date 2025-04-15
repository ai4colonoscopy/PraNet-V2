import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume

def convert_labels_to_one_hot_masks(batch_labels, num_classes):
    bs, H, W = batch_labels.shape
    one_hot_masks = torch.zeros((bs, num_classes, H, W), dtype=torch.uint8, device=batch_labels.device)
    # Perform one-hot encoding using the scatter method: set corresponding positions in one_hot_masks to 1 based on pixel values in batch_labels as indices.
    one_hot_masks.scatter_(1, batch_labels.unsqueeze(1), 1)
    inverted_masks = torch.logical_not(one_hot_masks).to(torch.uint8)
    
    return inverted_masks            
def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing,use_dual=args.dual)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("##########Disable test_save##########")
    if args.dual:
        logging.info("##########Using dual supervision##########")
    else:
        logging.info("##########Using single supervision##########")
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    bce_loss=nn.BCEWithLogitsLoss()

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.80
    # iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in tqdm(range(args.max_epochs)):
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            bg_mask=convert_labels_to_one_hot_masks(label_batch, args.num_classes)
            
            image_batch, label_batch, bg_mask = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor), bg_mask.type(torch.FloatTensor)

            image_batch, label_batch, bg_mask = image_batch.cuda(), label_batch.squeeze(1).cuda(),bg_mask.cuda()            
            
            P = model(image_batch, mode='train')

            if  not isinstance(P, list):
                P = [P]
            if epoch_num == 0 and i_batch == 0:
                n_outs = len(P)
                if args.dual:
                    n_outs = n_outs // 2
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                print('using ',args.supervision)
                print('ss:',ss)
            
            if args.dual:
                P_fg = P[:4]    # Extract foreground segmentation results, i.e., p1, p2, p3, p4
                P_bg = P[-4:]   # Extract background segmentation results, i.e., p1_bg, p2_bg, p3_bg, p4_bg
                loss = 0.0
                lc1, lc2, lc3 = 0.5, 0.7, 0.3
            
                for s in ss:
                    iout,ibg = 0.0,0.0
                    #print(s)
                    if(s==[]):
                        continue
                    for idx in range(len(s)):
                        iout += P_fg[s[idx]]
                        ibg += P_bg[s[idx]]
                    loss_ce = ce_loss(iout, label_batch[:].long())
                    loss_dice = dice_loss(iout, label_batch, softmax=True)
                    loss_bce=bce_loss(ibg, bg_mask[:])
                    loss += (lc1 * loss_ce + lc2 * loss_dice + lc3 * loss_bce) 
            else:
                loss = 0.0
                w_ce, w_dice = 0.3, 0.7
            
                for s in ss:
                    iout = 0.0
                    if(s==[]):
                        continue
                    for idx in range(len(s)):
                        iout += P[s[idx]]
                    loss_ce = ce_loss(iout, label_batch[:].long())
                    loss_dice = dice_loss(iout, label_batch, softmax=True)
                    loss += (w_ce * loss_ce + w_dice * loss_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
                
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss.item(), lr_))
        
        # save_mode_path = os.path.join(snapshot_path, 'last.pth')
        # torch.save(model.state_dict(), save_mode_path)
        
        save_interval = 50
        
        if epoch_num >= 0.5 * args.max_epochs:
            performance = inference(args, model, best_performance)

            if(best_performance <= performance):
                best_performance = performance
                save_mode_path = os.path.join(snapshot_path, 'best.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            
        # if (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break

    # writer.close()
    return "Training Finished!"
