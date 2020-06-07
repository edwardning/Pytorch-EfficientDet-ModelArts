# -*- coding: utf-8 -*-
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


#from model_service.pytorch_model_service import PTServingBaseService

import time
#from metric.metrics_manager import MetricsManager
#import log
#logger = log.getLogger(__name__)


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
        self.threshold = 0.5
        self.iou_threshold = 0.5
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

        self.model.load_state_dict(torch.load(self.model_path), strict=False)
        self.model.requires_grad_(False)
        self.model.eval()

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                ori_imgs, framed_imgs, framed_metas = preprocess(file_content, max_size=self.input_size)
                preprocessed_data[k] = [framed_imgs, framed_metas]
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        framed_imgs, framed_metas = data[self.input_image_key]
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

    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        print('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        print('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        print('postprocess time: ' + str(post_time_in_ms) + 'ms')


        print('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data


    '''
        def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data
    '''


if __name__ == '__main__':
    img_path = r'\img.png'
    model_path = r'\models\efficientdet-d0_9_930.pth'
    infer = ObjectDetectionService('', model_path)
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=512)

    result = infer._inference({"images": [framed_imgs, framed_metas]})

    print(result)



