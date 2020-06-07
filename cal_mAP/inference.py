# -*- coding: utf-8 -*-
"""
基于 https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py 修改得到的mAP计算脚本
"""
import json
import codecs
from collections import OrderedDict
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
import os
import cv2
import torch
import numpy as np
from torchvision.ops.boxes import batched_nms
from typing import Union
from tqdm import tqdm
import time
import pandas as pd
import xml.etree.ElementTree as ET
import pickle  # python2中使用的cPickle在python3中已改名为pickle
import argparse
try:
    import moxing as mox
    PLATFORM = 'ModelArts'
except:
    PLATFORM = 'Local'


def get_args():
    if torch.cuda.is_available():
        print('use GPU to eval')
    else:
        print('use CPU to eval')
    parser = argparse.ArgumentParser('cal mAP')
    parser.add_argument('-m', '--model_path', default=r'D:\coding\华为目标检测\PyTorch-YOLOv3-ModelArts\cal_mAP\models',
                        type=str, help='dir that contains models to eval')
    parser.add_argument('-i', '--img_path', default=r'D:\coding\trainval\VOC2007\test',
                        type=str, help='img path')
    parser.add_argument('-a', '--annopath', default=r'D:\coding\trainval\VOC2007\Annotations',
                        type=str, help='ann path that contains .xml files')
    parser.add_argument('-u', '--outpath', default=r'D:\coding\华为目标检测\PyTorch-YOLOv3-ModelArts\cal_mAP\out',
                        type=str, help='out path')

    args = parser.parse_args()
    return args


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagenames,
             classname,
             cachedir,
             basename,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, basename+'_annots.pkl')

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath[classname]

    if len(detfile) == 0:
        return [0.0], [0.0], 0.0

    splitlines = [x.strip().split(' ') for x in detfile]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    #ori_imgs = [cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), 1) for img_path in image_path]
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


class ObjectDetectionService():
    def __init__(self, model_name, model_path):
        # effdet
        self.model_name = model_name
        self.model_path = model_path
        self.input_image_key = 'images'
        self.anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.compound_coef = 0
        self.threshold = 0.2
        self.iou_threshold = 0.2
        self.obj_list = ['一次性快餐盒', '书籍纸张', '充电宝', '剩饭剩菜', '包', '垃圾桶', '塑料器皿', '塑料玩具',
           '塑料衣架', '大骨头', '干电池', '快递纸袋', '插头电线', '旧衣服', '易拉罐', '枕头',
           '果皮果肉', '毛绒玩具', '污损塑料', '污损用纸', '洗护用品', '烟蒂', '牙签', '玻璃器皿',
           '砧板', '筷子', '纸盒纸箱', '花盆', '茶叶渣', '菜帮菜叶', '蛋壳', '调料瓶', '软膏',
           '过期药物', '酒瓶', '金属厨具', '金属器皿', '金属食品罐', '锅', '陶瓷器皿', '鞋',
           '食用油桶', '饮料瓶', '鱼骨']
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = self.input_sizes[self.compound_coef]

        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list),
                                     ratios=self.anchor_ratios, scales=self.anchor_scales)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.requires_grad_(False)
        self.model.eval()

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                ori_imgs, framed_imgs, framed_metas = preprocess(file_content, max_size=self.input_size)
                preprocessed_data[k] = [framed_imgs, framed_metas]
        return preprocessed_data

    def _inference(self, img):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        _, framed_imgs, framed_metas = preprocess(img, max_size=self.input_size)
        if torch.cuda.is_available():
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            self.model = self.model.cuda()
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32).permute(0, 3, 1, 2)

        #if use_float16:
        #    model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              self.threshold, self.iou_threshold)

        out = invert_affine(framed_metas, out)
        result = OrderedDict()
        result['detection_classes'] = []
        result['detection_scores'] = []
        result['detection_boxes'] = []

        for i in range(len(out)):
            if len(out[i]['rois']) == 0:
                continue
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(np.int)
                result['detection_boxes'].append([x1, y1, x2, y2])
                obj = self.obj_list[out[i]['class_ids'][j]]
                result['detection_classes'].append(obj)
                score = float(out[i]['scores'][j])
                result['detection_scores'].append(score)

        return result


def prepare_data_on_modelarts(args):
    """
    将OBS上的数据拷贝到ModelArts中

    拷贝预训练参数文件

    默认使用ModelArts中的如下路径用于存储数据：
    1) /cache/model: 如果使用预训练模型，存储从OBS拷贝过来的预训练模型
    2）/cache/datasets: 存储从OBS拷贝过来的训练数据
    3）/cache/log: 存储训练日志和训练模型，并且在训练结束后，该目录下的内容会被全部拷贝到OBS
       /cache/log/logs/ 存放训练结果
       /cache/log/models/ 存放保存的模型
    """

    # 1) 拷贝图片
    cache_img = os.path.join(args.local_data_root, 'images')
    mox.file.copy_parallel(args.img_path, cache_img)
    args.img_path = cache_img
    # 2) 拷贝标签
    cache_ann = os.path.join(args.local_data_root, 'Annotations')
    mox.file.copy_parallel(args.annopath, cache_ann)
    args.annopath = cache_ann
    # 3) 拷贝模型
    cache_model = os.path.join(args.local_data_root, 'Annotations')
    mox.file.copy_parallel(args.model_path, cache_model)
    args.model_path = cache_model
    # 3) 创建临时输出目录
    cache_out = os.path.join(args.local_data_root, 'out')
    if not os.path.isdir(cache_out):
        os.mkdir(cache_out)
    args.outcache = cache_out
    return args


def evaulate(args, model, class_names):
    results = {}
    imgs = os.listdir(args.img_path)
    for class_name in class_names:
        results[class_name] = []
    imagesets = []
    for img in tqdm(imgs):
        file_name_prefix = img.split('.')[0]
        imagesets.append(file_name_prefix)
        detections = model._inference(os.path.join(args.img_path, img))

        for index, class_name in enumerate(detections['detection_classes']):
            box = detections['detection_boxes'][index]
            xmin = '%.1f' % box[0]  # 注意不要弄错坐标的顺序
            ymin = '%.1f' % box[1]
            xmax = '%.1f' % box[2]
            ymax = '%.1f' % box[3]
            line = (file_name_prefix + ' ' + '%.4f' % detections['detection_scores'][index] + ' ' + ' '.join(
                [xmin, ymin, xmax, ymax]) + '\n')
            results[class_name].append(line)
    return [results, imagesets]


if __name__ == '__main__':
    class_names = ['一次性快餐盒', '书籍纸张', '充电宝', '剩饭剩菜', '包', '垃圾桶', '塑料器皿', '塑料玩具',
                   '塑料衣架', '大骨头', '干电池', '快递纸袋', '插头电线', '旧衣服', '易拉罐', '枕头',
                   '果皮果肉', '毛绒玩具', '污损塑料', '污损用纸', '洗护用品', '烟蒂', '牙签', '玻璃器皿',
                   '砧板', '筷子', '纸盒纸箱', '花盆', '茶叶渣', '菜帮菜叶', '蛋壳', '调料瓶', '软膏',
                   '过期药物', '酒瓶', '金属厨具', '金属器皿', '金属食品罐', '锅', '陶瓷器皿', '鞋',
                   '食用油桶', '饮料瓶', '鱼骨']
    opt = get_args()
    if PLATFORM == 'ModelArts':
        opt = prepare_data_on_modelarts(opt)

    model_files = os.listdir(opt.model_path)
    models = []
    for m in model_files:
        if m.endswith('.pth'):
            models.append(os.path.join(opt.model_path, m))
    for model in models:
        base_name = os.path.split(model)[-1].split('.')[0]
        print('Evaluating with', base_name+'.pth')

        infer = ObjectDetectionService('', model)

        aps = []  # 保存各类ap
        eval_pd = pd.DataFrame(columns=['classname', 'AP', 'recall', 'precision'])  # 保存统计结果
        [results, imagesets] = evaulate(opt, infer, class_names)
        annopath = opt.annopath + '/{:s}.xml'
        for i, classname in enumerate(class_names):
            rec, prec, ap = voc_eval(results, annopath, imagesets, classname, opt.outpath, base_name,
                                     ovthresh=0.5, use_07_metric=False)
            aps.append(ap)
            eval_pd.loc[i] = [classname, '%.4f' % ap, '%.4f' % rec[-1], '%.4f' % prec[-1]]  # 插入一行
        eval_pd.loc[len(class_names)] = ['mAP', '%.4f' % np.mean(aps), None, None]

        if PLATFORM == 'ModelArts':
            eval_pd.to_excel(os.path.join(opt.outcache, base_name+'_mAP.xls'))
        else:
            eval_pd.to_excel(os.path.join(opt.outpath, base_name+'_mAP.xls'))
        print(eval_pd)

    if PLATFORM == 'ModelArts':
        mox.file.copy_parallel(opt.outcache, opt.outpath)







