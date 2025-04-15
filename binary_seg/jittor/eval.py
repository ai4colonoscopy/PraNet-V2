import os
import argparse
import tqdm
import sys

import numpy as np

from PIL import Image
from tabulate import tabulate

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.eval_functions import *
from utils.utils import *

def eval_for_testAllInOne(opt,pred,gt):
    pred_mask = np.array(pred)
    gt_mask = np.array(gt)
    
    Thresholds = np.linspace(1, 0, 256)
    
    assert pred_mask.shape == gt_mask.shape

    t_mask = gt_mask.astype(np.float64)
    gt_mask = (gt_mask > 0.5).astype(np.float64)

    pred_mask = pred_mask.astype(np.float64) / 255
    
    # Initialize metric calculation
    Smeasure = StructureMeasure(pred_mask, gt_mask)
    wFmeasure = original_WFb(pred_mask, gt_mask)
    MAE = np.mean(np.abs(gt_mask - pred_mask))

    threshold_E = np.zeros(len(Thresholds))
    threshold_F = np.zeros(len(Thresholds))
    threshold_Pr = np.zeros(len(Thresholds))
    threshold_Rec = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    threshold_Spe = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))
    
    for j, threshold in enumerate(Thresholds):
        threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)
        Bi_pred = np.zeros_like(pred_mask)
        Bi_pred[pred_mask >= threshold] = 1
        threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
        
    meanDic = np.mean(threshold_Dic)
    meanIoU = np.mean(threshold_Iou)
    meanEm = np.mean(threshold_E)
    # maxEm = np.max(threshold_E)
    # meanSen = np.mean(threshold_Rec)
    # maxSen = np.max(threshold_Rec)
    # meanSpe = np.mean(threshold_Spe)
    # maxSpe = np.max(threshold_Spe)
    
    mae = np.mean(MAE)
    Sm = np.mean(Smeasure)
    wFm = np.mean(wFmeasure)

    results = []
    for metric in opt["metrics"]:
        results.append(eval(metric))
    return results
    
    
    


def evaluate(opt):
    if os.path.isdir(opt["result_path"]) is False:
        os.makedirs(opt["result_path"])

    method = os.path.split(opt["pred_root"])[-1]
    Thresholds = np.linspace(1, 0, 256)
    headers = opt["metrics"] #['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen', 'meanSpe', 'maxSpe']
    results = []
    
    print('#' * 20, 'Start Evaluation', '#' * 20)
    datasets = tqdm.tqdm(opt["datasets"], desc='Expr - ' + method, total=len(
        opt["datasets"]), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')

    for dataset in datasets:
        pred_root = os.path.join(opt["pred_root"], dataset)
        gt_root = os.path.join(opt["gt_root"], dataset, 'masks')

        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        # threshold_Precision = np.zeros((len(preds), len(Thresholds)))
        # threshold_Recall = np.zeros((len(preds), len(Thresholds)))
        threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
        threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))

        samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(
            preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        

        for i, sample in samples:
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))
            
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            
            assert pred_mask.shape == gt_mask.shape, f"Shape mismatch: {pred_mask.shape} != {gt_mask.shape}"

            gt_mask = gt_mask.astype(np.float64)
            gt_mask = (gt_mask > 0.5).astype(np.float64)

            pred_mask = pred_mask.astype(np.float64) / 255

            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))

            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
            
            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []

        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        # result.extend([meanDic, meanIoU, wFm, Sm, meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe])
        # results.append([dataset, *result])
        
        out = []
        for metric in opt["metrics"]:
            out.append(eval(metric))

        result.extend(out)
        results.append([dataset, *result])

        csv = os.path.join(opt["result_path"], 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(['method', *headers]) + '\n')

        out_str = method + ','
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'

        csv.write(out_str)
        csv.close()
    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")

    print(tab)
    print("#"*20, "End Evaluation", "#"*20)
        
    return tab

if __name__ == "__main__":
    # ## all in one ###
    eval_config={
    "gt_root": "../data/TestDataset",
    "pred_root": "...",
    "result_path": "eval_results",
    "datasets": ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'ETIS-LaribPolypDB'],
    "metrics": ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae'],
    }
    pred_roots = ['results/PraNet-V1', 'results/PVT-PraNet-V1', 'results/PraNet-V2', 'results/PVT-PraNet-V2']
    for pred_root in pred_roots:
        eval_config["pred_root"] = pred_root
        evaluate(eval_config)