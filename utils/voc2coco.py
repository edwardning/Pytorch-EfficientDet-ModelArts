#!/usr/bin/python
'''
target: change voc datasets to coco datasets

1)typical format of voc datasets:
VOC2007/
    Annotations/
        0.xml
        1.xml
        ...
    ImageSets/
        Main/
            train.txt
            test.txt
            val.txt
            trainval.txt
    JPEGImages/
        0.jpg
        1.jpg
        ...
2)typical format of coco_datasets:
coco/
    annotations/
        instances_train2017.json
        instances_val2017.json
    train2017/
        0.jpg
        ...
    val2017
        1.jpg
        ...
'''
import os
import json
from tqdm import tqdm
import random
import shutil
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1
# PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {'一次性快餐盒': 0, '书籍纸张': 1, '充电宝': 2, '剩饭剩菜': 3, '包': 4,
                         '垃圾桶': 5, '塑料器皿': 6, '塑料玩具': 7, '塑料衣架': 8, '大骨头': 9,
                         '干电池': 10, '快递纸袋': 11, '插头电线': 12, '旧衣服': 13, '易拉罐': 14,
                         '枕头': 15, '果皮果肉': 16, '毛绒玩具': 17, '污损塑料': 18, '污损用纸': 19,
                         '洗护用品': 20, '烟蒂': 21, '牙签': 22, '玻璃器皿': 23, '砧板': 24,
                         '筷子': 25, '纸盒纸箱': 26, '花盆': 27, '茶叶渣': 28, '菜帮菜叶': 29,
                         '蛋壳': 30, '调料瓶': 31, '软膏': 32, '过期药物': 33, '酒瓶': 34,
                         '金属厨具': 35, '金属器皿': 36, '金属食品罐': 37, '锅': 38, '陶瓷器皿': 39,
                         '鞋': 40, '食用油桶': 41, '饮料瓶': 42, '鱼骨': 43}
# note that, PRE_DEFINE_CATEGORIES should be ONE-INDEXED!!!

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        # print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()

        ## The filename must be a number
        image_id = get_filename_as_int(line)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': line.replace('.xml', '.jpg'), 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


def voc2coco(data_path, coco_path, val_ratio):
    if not os.path.exists(coco_path):
        os.makedirs(coco_path)
    NUM = 1
    ano_path = os.path.join(data_path, 'Annotations')
    img_path = os.path.join(data_path, 'JPEGImages')
    txt_path = os.path.join(data_path, 'ImageSets')

    files = os.listdir(ano_path)

    print('rename files...')
    for file in tqdm(files):
        os.rename(os.path.join(ano_path, file), os.path.join(ano_path, str(NUM) + '.xml'))
        os.rename(os.path.join(img_path, file).replace('.xml', '.jpg'),
                  os.path.join(img_path, str(NUM) + '.jpg'))
        NUM += 1

    print('split images...')
    files = os.listdir(ano_path)
    val = random.sample(files, int(len(files) * val_ratio))
    train = [x for x in files if not x in val]
    with open(os.path.join(txt_path, 'val.txt'), 'w') as fv:
        for v in val:
            fv.write(str(v) + '\n')
            val_path = os.path.join(coco_path, 'val')
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            shutil.move(os.path.join(img_path, v.replace('.xml', '.jpg')),
                        os.path.join(val_path, v.replace('.xml', '.jpg')))
    with open(os.path.join(txt_path, 'train.txt'), 'w') as ft:
        for t in tqdm(train):
            ft.write(str(t) + '\n')
            train_path = os.path.join(coco_path, 'train')
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            shutil.move(os.path.join(img_path, t.replace('.xml', '.jpg')),
                        os.path.join(train_path, t.replace('.xml', '.jpg')))

    print('create json files...')
    if not os.path.exists(os.path.join(coco_path, 'annotations')):
        os.makedirs(os.path.join(coco_path, 'annotations'))
    convert(os.path.join(txt_path, 'val.txt'), ano_path, os.path.join(coco_path,'annotations\instances_val.json'))
    convert(os.path.join(txt_path, 'train.txt'), ano_path, os.path.join(coco_path, 'annotations\instances_train.json'))


if __name__ == '__main__':
    # 为保证文件名和json文件中一致，voc_path 目录下的文件会被重命名，请注意备份
    voc_path = r'D:\coding\trainval\VOC2007'
    coco_path = r'D:\coding\PEM'
    voc2coco(voc_path, coco_path, 0.2)
