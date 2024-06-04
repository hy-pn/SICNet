from typing import Type
from sklearn import set_config
from utils.utils import SCDD_eval_all_test
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import argparse
import numpy as np
from datasets import RS_ST as RS
from torch.utils.data import DataLoader
import torch
class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        
    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='', help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default='', help='directory to output masks')
        
        self.initialized = True
        return parser
    
    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt

def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    
    test_set = RS.Data_test_eval(opt.test_dir)
    pred_set = RS.Data_test_eval(opt.pred_dir)
    x=np.max(pred_set.imgs_A_label[0])
    test_loader = DataLoader(test_set)
    pred_loader = DataLoader(pred_set)
    preds_all = []
    tests_all = []
    for  i in range(test_set.len):
        imgs_A_label=torch.tensor(test_set.imgs_A_label[i])
        imgs_B_label=torch.tensor(test_set.imgs_B_label[i])
        x=torch.max(imgs_A_label)
        imgs_A_label = (imgs_A_label).int()
        imgs_B_label = (imgs_B_label).int()
        preds_all.append(imgs_A_label)
        preds_all.append(imgs_B_label)
    for i in range(pred_set.len):
        imgs_A_label=torch.tensor(pred_set.imgs_A_label[i])
        imgs_B_label=torch.tensor(pred_set.imgs_B_label[i])
        tests_all.append(imgs_A_label)
        tests_all.append(imgs_B_label)
    Score, IoU_mean, IoU_fg, Sek, Fscd, Fscd1, changepre, recall= SCDD_eval_all_test(preds_all, tests_all, RS.num_classes)
    print('Score: %.2f mIoU: %.2f IoU: %.2f Sek: %.2f Fscd: %.2f  F1: %.2f  changepre: %.2f  recall: %.2f'\
    %(Score*100, IoU_mean*100, IoU_fg*100, Sek*100 , Fscd*100, Fscd1*100, changepre*100, recall*100))
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)
def main1():
    begin_time = time.time()
    opt = PredOptions().parse()
    
    test_set = RS.Data_test_eval(opt.test_dir)
    pred_set = RS.Data_test_eval(opt.pred_dir)
    x=np.max(pred_set.imgs_A_label[0])
    test_loader = DataLoader(test_set)
    pred_loader = DataLoader(pred_set)
    preds_all = []
    tests_all = []
    
    for  i in range(test_set.len):
        cdclass = 0
        img_A_label=torch.tensor(test_set.imgs_A_label[i])
        img_B_label=torch.tensor(test_set.imgs_B_label[i])
        img_label = torch.zeros(img_A_label.shape[0], img_A_label.shape[1])
        for first in range(7):
            for second in range(7):
                img_label[img_A_label == first and img_B_label == second] = cdclass
                cdclass = cdclass + 1
        
        img_A_label = (img_A_label).int()
        img_B_label = (img_B_label).int()
        preds_all.append(img_label)
    for i in range(pred_set.len):
        cdclass = 0
        imgs_A_label=torch.tensor(pred_set.imgs_A_label[i])
        imgs_B_label=torch.tensor(pred_set.imgs_B_label[i])
        pre_label = torch.zeros(img_A_label.shape[0], img_A_label.shape[1])
        for first in range(7):
            for second in range(7):
                pre_label[img_A_label == first and img_B_label == second] = cdclass
                cdclass = cdclass + 1
        tests_all.append(pre_label)
    score, IoU_mean, Sek , kappa, changepre= SCDD_eval_all_test(preds_all, tests_all, RS.num_classes)
    print('Score: %.2f mIoU: %.2f Sek: %.2f kappa_n0: %.2f  changepre: %.2f'\
    %(score*100, IoU_mean*100, Sek*100 , kappa*100, changepre*100))
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)
if __name__ == '__main__':
    main()