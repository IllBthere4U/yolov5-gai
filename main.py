# import os
# import sys
# from pathlib import Path
#
# import cv2
# import torch
#
# # 获取当前文件的路径
# FILE = Path(__file__).resolve()
# # 获取YOLOv5的根目录
# ROOT = FILE.parents[0]
# # 如果ROOT不在系统路径中，则添加到系统路径
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# # 将ROOT路径转为相对路径
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
#
# # 导入相关模块
# from models.common import DetectMultiBackend
# from utils.augmentations import letterbox
# from utils.general import (non_max_suppression, scale_coords)
# from utils.torch_utils import select_device
# import numpy as np
#
# # 加载模型的函数
# def load_model(weights='./best.pt',
#                data=Path(__file__).resolve().parents[0] / 'data/coco128.yaml',
#                device='',
#                half=False,
#                dnn=False):
#     # 选择设备，如GPU或CPU
#     device = select_device(device)
#     # 加载模型
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
#     stride = model.stride
#     names = model.names
#     pt = model.pt
#     return model, stride, names, pt
#
# # 运行模型并返回检测结果的函数
# def run(model, img, stride, pt, imgsz=(640, 640), conf_thres=0.05, iou_thres=0.10, max_det=1000, device='', classes=None, agnostic_nms=False, augment=False, half=False):
#     cal_detect = []
#     device = select_device(device)
#     # 获取类别名称
#     names = model.module.names if hasattr(model, 'module') else model.names
#
#     # 图像预处理：调整尺寸
#     im = letterbox(img, imgsz, stride, pt)[0]
#
#     # 图像格式转换
#     im = im.transpose((2, 0, 1))[::-1]
#     im = np.ascontiguousarray(im)
#
#     # 图像预处理：归一化等操作
#     im = torch.from_numpy(im).to(device)
#     im = im.half() if half else im.float()
#     im /= 255.0
#     if len(im.shape) == 3:
#         im = im[None]
#
#     # 对图像进行目标检测
#     pred = model(im, augment=augment)
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#
#     # 处理检测结果，如调整坐标
#     for i, det in enumerate(pred):
#         if len(det):
#             det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 c = int(cls)
#                 label = f'{names[c]}'
#                 cal_detect.append([label, xyxy, str(float(conf))[:5]])
#
#     return cal_detect
#
# # 主函数
# if __name__ == "__main__":
#     # 加载模型
#     model, stride, names, pt = load_model()
#
#     test_path = './test'  # 定义test文件夹的路径
#     # 遍历test文件夹中的每一张图片
#     for img_path in os.listdir(test_path):
#         full_path = os.path.join(test_path, img_path)
#         if os.path.isfile(full_path):
#             img = cv2.imread(full_path)
#             # 获取每张图片的检测结果
#             detections = run(model, img, stride, pt)
#
#             # 将检测结果绘制到图片上
#             for detection in detections:
#                 label, (x1, y1, x2, y2), conf = detection
#                 cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 cv2.putText(img, f'{label} {conf}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#             # 使用OpenCV展示带有检测结果的图片
#             img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
#             cv2.imshow('Detection', img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()


import os
from pathlib import Path
import cv2
import torch
import sys
import json
import numpy as np
from datetime import datetime

# 获取当前执行的文件的绝对路径
FILE = Path(__file__).resolve()
# 获取YOLOv5的根目录
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

def load_model(weights='./best.pt',
               data=FILE.parents[0] / 'data/coco128.yaml',
               device='',
               half=False,
               dnn=False):
    """
    加载目标检测模型。
    
    参数:
    - weights: 模型权重的路径
    - data: 数据集的配置文件路径
    - device: 设备选择 ("0": GPU, "cpu": CPU)
    - half: 是否使用半精度浮点数
    - dnn: 是否使用DNN模块
    
    返回:
    - 模型对象, stride, 类别名称, pt
    """
    # 优先选择GPU，如果GPU不可用则选择CPU
    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride = model.stride
    names = model.names
    pt = model.pt
    return model, stride, names, pt

def run(model, img, stride, pt, imgsz=(640, 640), conf_thres=0.5, iou_thres=0.10, max_det=1000, device='', classes=None, agnostic_nms=False, augment=False, half=False):
    """
    使用加载的模型对图像进行目标检测。
    
    参数:
    - model: 加载的模型
    - img: 输入图像
    ... 其他模型和检测参数 ...
    
    返回:
    - 检测结果列表
    """
    cal_detect = []
    device = select_device(device)
    names = model.module.names if hasattr(model, 'module') else model.names
    im = letterbox(img, imgsz, stride, pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]
    pred = model(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]}'
                # 检查xyxy的数据类型并进行相应的处理
                if isinstance(xyxy, torch.Tensor):
                    xyxy = xyxy.cpu().numpy().tolist()
                conf_value = float(conf.item()) if isinstance(conf, torch.Tensor) else float(conf)
                cal_detect.append([label, xyxy, str(conf_value)[:5]])
    for item in cal_detect:
        for subitem in item:
            if isinstance(subitem, torch.Tensor):
                print("Found a tensor in cal_detect:", item)

    return cal_detect

def save_json(results, output_folder="./json"):
    """
    将检测结果保存为.json文件。
    
    参数:
    - results: 检测结果列表
    - output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "detections.json")
    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)


def main():
    model, stride, names, pt = load_model()
    test_path = './test'

    for img_path in os.listdir(test_path):
        full_path = os.path.join(test_path, img_path)
        extension = os.path.splitext(img_path)[1].lower()

        if os.path.isfile(full_path) and extension in [".bmp", ".png", ".jpg", ".jpeg"]:
            img = cv2.imread(full_path)
            detections = run(model, img, stride, pt)

            # 按x1值从左到右排序检测框
            detections.sort(key=lambda x: x[1][0])

            # 创建保存截取图像的文件夹
            output_folder = f'./clip/{datetime.now().strftime("%Y-%m-%d")}'
            os.makedirs(output_folder, exist_ok=True)

            for idx, detection in enumerate(detections):
                label, (x1, y1, x2, y2), conf = detection

                # 截取并保存图像
                cropped = img[int(y1):int(y2), int(x1):int(x2)]
                cropped_filename = f"{os.path.splitext(img_path)[0]}_{idx + 1}.png"
                cropped_path = os.path.join(output_folder, cropped_filename)
                cv2.imwrite(cropped_path, cropped)

            # 在原图上画矩形并显示（在裁剪之后执行）
            for detection in detections:
                label, (x1, y1, x2, y2), conf = detection
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)


            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            # cv2.imshow('Detection', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 转化.bmp到.png
            if extension != ".png":
                png_path = os.path.join(test_path, f"{os.path.splitext(img_path)[0]}.png")
                cv2.imwrite(png_path, img)
                os.remove(full_path)


if __name__ == "__main__":
    main()
