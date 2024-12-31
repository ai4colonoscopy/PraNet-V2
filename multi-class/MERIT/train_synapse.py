import argparse
import logging
import os
import time
import random
import numpy as np
from datetime import datetime
import pytz

import torch
import torch.backends.cudnn as cudnn

from lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded, MERIT_Cascaded_dual, MERIT_Parallel_dual

from trainer import trainer_synapse

# from torchsummaryX import summary
from ptflops import get_model_complexity_info




parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='/defaultShare/archive/zhuzixuan/cascade_dataset/synapse/train_npz_new', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='/defaultShare/archive/zhuzixuan/cascade_dataset/synapse/test_vol_h5_new', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate') #0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input') #224
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--dual',action='store_true', help='dual supervision or single supervision')
parser.add_argument('--merit_type',default='MERIT_Cascaded', help='MERIT_Cascaded or MERIT_Parallel')
parser.add_argument('--test_save_dir', default=None, help='saving prediction as nii!')


args = parser.parse_args()


if __name__ == "__main__":
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
    
    
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    
    if args.merit_type == 'MERIT_Cascaded':
        args.exp = 'MERIT_Cascaded_Small_loss_MUTATION_w3_7_' + str(args.img_size)
        print("########## Using MERIT_Cascaded ##########")
    if args.merit_type == 'MERIT_Parallel':
        args.exp = 'MERIT_Parallel_Small_loss_MUTATION_w3_7_' + str(args.img_size)
        print("########## Using MERIT_Parallel ##########")
    current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H")
    print("The current time is", current_time)

    if args.dual:
        args.exp = 'Dual_' + args.exp

    args.exp='run:'+current_time+'_'+args.exp
    snapshot_path = "model_pth/{}/{}".format(dataset_name, args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    if args.dual:
        if args.merit_type == 'MERIT_Parallel':
            net=MERIT_Parallel_dual(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
        if args.merit_type == 'MERIT_Cascaded':
            net=MERIT_Cascaded_dual(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    else:
        if args.merit_type == 'MERIT_Parallel':
            net = MERIT_Parallel(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
        if args.merit_type == 'MERIT_Cascaded':
            net = MERIT_Cascaded(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()

    # net = MERIT_Cascaded(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')

    
    print('Model %s created, param count: %d' %
                     (args.merit_type, sum([m.numel() for m in net.parameters()])))

    net = net.cuda()
   
    macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)
