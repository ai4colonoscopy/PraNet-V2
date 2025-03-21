import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time

from utils.utils import test_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from lib.networks import MaxViT, MaxViT4Out, MaxViT_CASCADE, MERIT_Parallel, MERIT_Cascaded, MERIT_Cascaded_dual, MERIT_Parallel_dual
        
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
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=12, help="batch size")
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
    parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
    parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
    parser.add_argument('--dual',action='store_true', help='dual supervision or single supervision')
    parser.add_argument('--merit_type',default='MERIT_Cascaded', help='MERIT_Cascaded or MERIT_Parallel')
                
    args = parser.parse_args()

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

    # config_vit.n_classes = args.num_classes
    # config_vit.n_skip = args.n_skip

    args.is_pretrain = True
    
    
    if args.merit_type == 'MERIT_Cascaded':
        args.exp = 'MERIT_Cascaded_Small_loss_MUTATION_w3_7_' + str(args.img_size)
        print("########## Using MERIT_Cascaded ##########")
    if args.merit_type == 'MERIT_Parallel':
        args.exp = 'MERIT_Parallel_Small_loss_MUTATION_w3_7_' + str(args.img_size)
        print("########## Using MERIT_Parallel ##########")
    current_time = time.strftime("%H%M%S")
    print("The current time is", current_time)
    

    if args.dual:
        args.exp = 'Dual_' + args.exp
    args.exp='run:'+current_time+'_'+args.exp
    snapshot_path = "{}/{}".format(args.save_path, args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    
    
    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # logging.info(snapshot_name)
    
    snapshot_path = '' # TODO：Replace with the path to the model you want to test
    
    if args.is_savefig:
        test_save_path = os.path.join(os.path.dirname(snapshot_path), 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    
    if args.dual:
        logging.info("########## Using Dual Supervision ##########")
        net=MERIT_Cascaded_dual(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    else:
        logging.info("########## Using Single Supervision ##########")
        net=MERIT_Cascaded(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()


    net.load_state_dict(torch.load(snapshot_path))

    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    results = inference(args, net, testloader, test_save_path)


