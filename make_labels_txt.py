'''
这个函数是为了将annotations中的.xml文件信息中的类别，bbox的信息转换成labels文件夹下的.txt文件
'''
import xml.etree.ElementTree as ET
import os
import cv2

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']


DATASET_PATH = r"D:\my_dataset\VOC2012"
## 对角两点转换成中心点＋宽和高  并进行归一化
def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]

    ##横纵坐标中心
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    ## 宽度和高度
    w = box[1] - box[0]
    h = box[3] - box[2]

   ## 四个变量进行归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    """把图像image_id的xml文件转换为目标检测的labels文件(txt)
    其中包含物体的类别，bbox的中心点坐标以及bbox的宽、高
    并将四个物理量归一化"""
    DATASET_PATH = r"D:\my_dataset\VOC2012"
    in_file = open(DATASET_PATH +"\\"+ 'Annotations/%s' % (image_id))
    image_id = image_id.split('.')[0]  ## 2007——000027.xml取前面的数字

    out_file = open('DATASET_PATH/labels/%s.txt' % (image_id), 'w')  ## 当前目录中创建labels

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
### 一副图片里面可能有多个标注信息
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        ## 将对角线上的两个点转换成中心点加宽和高的形式
        bb = convert((w, h), points)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    DATASET_PATH = r"D:\my_dataset\VOC2012"
    path_list=os.path.join(DATASET_PATH,"Annotations")
    filenames = os.listdir(path_list)
    for file in filenames:
        convert_annotation(file)
make_label_txt()