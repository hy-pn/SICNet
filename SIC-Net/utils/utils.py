import os
import math
import random
import numpy as np
from scipy import stats
from utils import eval_segm as seg_acc
from sklearn.metrics import confusion_matrix

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new

# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def ImageValStretch2D(img):
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)

def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf

def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
    
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist_landsat(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += confusion_matrix(label.flatten(), image.flatten(), labels=[0, 1, 2, 3, 4])
    return hist
def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += confusion_matrix(label.flatten(), image.flatten(), labels=[0, 1, 2, 3, 4,5,6])
    return hist
def get_hist7(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += confusion_matrix(label.flatten(), image.flatten(), labels=[0, 1, 2, 3, 4,5,6,7])
    return hist
def get_hist6(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += confusion_matrix(label.flatten(), image.flatten(), labels=[0, 1, 2, 3, 4,5])
    return hist
def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

def calculate_f1_score(confusion_matrix, class_index):
    true_positives = confusion_matrix[class_index, class_index]
    false_positives = np.sum(confusion_matrix[:, class_index]) - true_positives
    false_negatives = np.sum(confusion_matrix[class_index, :]) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def calculate_class_f1_scores(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    f1_scores = []

    for class_index in range(num_classes):
        f1_score = calculate_f1_score(confusion_matrix, class_index)
        f1_scores.append(f1_score)

    return f1_scores
def SCDD_eval_all_test(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target9 must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    hist_ss = hist_fg
    kappa = cal_kappa(hist_ss)
    changepre = np.diag(hist).sum() / c2hist.sum()
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    precision = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[0][1])
    recall = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision = c2hist[1][1]/(c2hist[1][1]+c2hist[0][1])
    recall = c2hist[1][1]/(c2hist[1][1]+c2hist[1][0])
    F1_cd = (2*precision*recall)/(precision+recall)
    F1_ss = calculate_class_f1_scores(hist_ss)
    f1_mean_ss = np.mean(F1_ss)
    return Score, IoU_mean, IoU_fg, Sek, Fscd, F1_cd, changepre, recall,kappa,F1_ss,f1_mean_ss,precision
def SCDD_eval_all_test_24(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target9 must be the same"
        hist += get_hist_landsat(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    hist_ss = hist_fg
    kappa = cal_kappa(hist_ss)
    changepre = np.diag(hist).sum() / c2hist.sum()
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    precision = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[0][1])
    recall = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision1 = c2hist[1][1]/(c2hist[1][1]+c2hist[0][1])
    recall1 = c2hist[1][1]/(c2hist[1][1]+c2hist[1][0])
    F1_cd = (2*precision1*recall1)/(precision1+recall1)
    F1_ss = calculate_class_f1_scores(hist_ss)
    f1_mean_ss = np.mean(F1_ss)
    return Score, IoU_mean, IoU_fg, Sek, Fscd, F1_cd, changepre, recall1,kappa,F1_ss,f1_mean_ss,precision1
def SCDD_eval_all_test_landsat(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target9 must be the same"
        hist += get_hist_landsat(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    hist_ss = hist_fg
    kappa = cal_kappa(hist_ss)
    changepre = np.diag(hist).sum() / c2hist.sum()
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    precision = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[0][1])
    recall = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision = c2hist[1][1]/(c2hist[1][1]+c2hist[0][1])
    recall = c2hist[1][1]/(c2hist[1][1]+c2hist[1][0])
    F1_cd = (2*precision*recall)/(precision+recall)
    F1_ss = calculate_class_f1_scores(hist_ss)
    f1_mean_ss = np.mean(F1_ss)
    return Score, IoU_mean, IoU_fg, Sek, Fscd, F1_cd, changepre, recall,kappa,F1_ss,f1_mean_ss
def SCDD_eval_all_msscd(preds1,preds2, labels1,labels2, num_class1,num_class2):
    # hist1 = np.zeros((num_class1, num_class1))
    hist2 = np.zeros((num_class2, num_class2))
    # for pred, label in zip(preds1, labels1):
    #     infer_array = np.array(pred)
    #     unique_set = set(np.unique(infer_array))
    #     assert unique_set.issubset(set([0, 1])), "unrecognized label number"
    #     label_array = np.array(label)
    #     assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
    #     hist1 += get_hist(infer_array, label_array, num_class1)
    for pred, label in zip(preds2, labels2):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6,7])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist2 += get_hist7(infer_array, label_array, num_class2)
    # hist_fg = hist1[1:, 1:]
    # c2hist = np.zeros((2, 2))
    # c2hist[0][0] = hist1[0][0]
    # c2hist[0][1] = hist1.sum(1)[0] - hist1[0][0]
    # c2hist[1][0] = hist1.sum(0)[0] - hist1[0][0]
    # c2hist[1][1] = hist_fg.sum()
    # hist_n0 = hist1.copy()
    # hist_n0[0][0] = 0
    # kappa_n0 = cal_kappa(hist_n0)
    # iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    # IoU_fg = iu[1]

    # IoU_mean = (iu[0] + iu[1]) / 2

    hist_fg2 = hist2[1:, 1:]
    c2hist2 = np.zeros((2, 2))
    c2hist2[0][0] = hist2[0][0]
    c2hist2[0][1] = hist2.sum(1)[0] - hist2[0][0]
    c2hist2[1][0] = hist2.sum(0)[0] - hist2[0][0]
    c2hist2[1][1] = hist_fg2.sum()
    hist_n02 = hist2.copy()
    hist_n02[0][0] = 0
    kappa_n02 = cal_kappa(hist_n02)
    iu2 = np.diag(c2hist2) / (c2hist2.sum(1) + c2hist2.sum(0) - np.diag(c2hist2))
    IoU_fg2 = iu2[1]
    IoU_mean2 = (iu2[0] + iu2[1]) / 2

    kappa_0 = kappa_n02
    IoU_fg0 = IoU_fg2
    Sek = (kappa_0 * math.exp(IoU_fg0)) / math.e
    IoU_mean = IoU_mean2
    Score = 0.3*IoU_mean + 0.7*Sek

    precision = np.diag(hist_n02).sum()/(c2hist2[1][1]+c2hist2[0][1])
    recall = np.diag(hist_n02).sum()/(c2hist2[1][1]+c2hist2[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision = c2hist2[1][1]/(c2hist2[1][1]+c2hist2[0][1])
    recall = c2hist2[1][1]/(c2hist2[1][1]+c2hist2[1][0])
    F1 = (2*precision*recall)/(precision+recall)
    kappa = cal_kappa(hist_fg2)
    return Score, IoU_mean, Sek,Fscd,F1,kappa

def SCDD_eval_all_test_msscd(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6,7])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target9 must be the same"
        hist += get_hist7(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    kappa = cal_kappa(hist_fg)
    changepre = np.diag(hist).sum() / c2hist.sum(0).sum()
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    precision = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[0][1])
    recall = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision = c2hist[1][1]/(c2hist[1][1]+c2hist[0][1])
    recall = c2hist[1][1]/(c2hist[1][1]+c2hist[1][0])
    F1 = (2*precision*recall)/(precision+recall)
    
    return Score, IoU_mean, IoU_fg, Sek, Fscd, F1, changepre, precision,recall,kappa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix', cmap=plt.cm.Blues,font = None):
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, fontproperties=font)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2%'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', bbox_inches='tight', pad_inches=0)
    return 

def SCDD_eval_all_test_msscd_visual(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6,7])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target9 must be the same"
        hist += get_hist7(infer_array, label_array, num_class)
    # 可视化混淆矩阵，设置颜色为蓝色
    # font = FontProperties(fname="/home/HDD1/experiment/字魂正文宋楷(商用需授权).ttf")
    # # conf_matrix = np.delete(hist, [0,3, 7], axis=0)
    # # conf_matrix = np.delete(conf_matrix, [0,3, 7], axis=1)
    # hist_fg = hist[1:, 1:]
    # c2hist = np.zeros((2, 2))
    # c2hist[0][0] = hist[0][0]
    # c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    # c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    # c2hist[1][1] = hist_fg.sum()
    # conf_matrix = c2hist/c2hist.sum()
    # conf_matrix = conf_matrix/conf_matrix.sum()
    # # classes = ['未变化', '建筑', '彩钢房','推土区','道路','在建道路']
    # classes = ['2', '3','4','5','6']
    # # classes = ['unchange', 'change']


    # plot_confusion_matrix(conf_matrix, classes, cmap=plt.cm.Blues,font= font)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    kappa = cal_kappa(hist_fg)
    changepre = np.diag(hist).sum() / c2hist.sum(0).sum()
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    precision = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[0][1])
    recall = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision = c2hist[1][1]/(c2hist[1][1]+c2hist[0][1])
    recall = c2hist[1][1]/(c2hist[1][1]+c2hist[1][0])
    F1 = (2*precision*recall)/(precision+recall)
    
    return Score, IoU_mean, IoU_fg, Sek, Fscd, F1, changepre, precision,recall,kappa
def SCDD_eval_all_landsat(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist_landsat(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    return Score, IoU_mean, Sek
def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    return Score, IoU_mean, Sek
def SCDD_eval_all2(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist6(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    precision = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[0][1])
    recall = np.diag(hist_n0).sum()/(c2hist[1][1]+c2hist[1][0])
    Fscd = (2*precision*recall)/(precision+recall)
    precision = c2hist[1][1]/(c2hist[1][1]+c2hist[0][1])
    recall = c2hist[1][1]/(c2hist[1][1]+c2hist[1][0])
    F1 = (2*precision*recall)/(precision+recall)
    kappa = cal_kappa(hist_fg)
    return Score, IoU_mean, Sek,Fscd,F1,kappa

def SCDD_eval(pred, label, num_class):
    infer_array = np.array(pred)
    unique_set = set(np.unique(infer_array))
    assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
    label_array = np.array(label)
    assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
    hist = get_hist(infer_array, label_array, num_class)
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3*IoU_mean + 0.7*Sek
    return Score, IoU_mean, Sek

def FWIoU(pred, label, bn_mode=False, ignore_zero=False):
    if bn_mode:
        pred = (pred>= 0.5)
        label = (label>= 0.5)
    elif ignore_zero:
        pred = pred-1
        label = label-1
    FWIoU = seg_acc.frequency_weighted_IU(pred, label)
    return FWIoU

def binary_accuracy(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)

    return (area_intersection, area_union)

def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)


    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist)