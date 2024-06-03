import numpy as np
import torch
import cv2
import base64
import yaml
from pathlib import Path

import bentoml
from bentoml.io import Image, JSON

yolov8s_runner = bentoml.pytorch.get("yolov8s_model:latest").to_runner()
svc = bentoml.Service("yolov8s_svc", runners=[yolov8s_runner])

from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.data.augment import LetterBox


class Model:
    def __init__(self):
        with open('./ultralytics/datasets/coco.yaml', errors='ignore') as f:
            self.names = yaml.safe_load(f)['names']


def encode_image(input_img):
    ratio = 3  # 0~9
    encode_param = [cv2.IMWRITE_PNG_COMPRESSION, ratio]
    encoded_img = base64.b64encode(cv2.imencode(".png", input_img, encode_param)[1])

    return encoded_img.decode("utf8")


def pre_processing(f: Image,
                   img_size: int = 640,
                   stride: int = 32):
    img_origin = np.array(f)
    img = LetterBox(img_size, True, stride=stride)(image=img_origin)
    img = img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    img = torch.from_numpy(img).cuda()
    img = img.float() / 255.
    img = img[None]
    return img_origin, img


def post_processing(img_origin: np.array,
                    img_tensor: torch.Tensor,
                    out: torch.Tensor):
    predictor = DetectionPredictor()
    predictor.model = Model()
    preds = predictor.postprocess(out, img_tensor, img_origin)
    _ = predictor.write_results(0, preds, ((Path('dummy'), img_tensor, img_origin)))
    return preds[0].boxes, cv2.cvtColor(predictor.plotted_img, cv2.COLOR_RGB2BGR)


@svc.api(input=Image(),
         output=JSON())
def predict(f: Image):
    img_origin, img_tensor = pre_processing(f=f)
    out = yolov8s_runner.run(img_tensor)
    out_bbox_info, out_img = post_processing(img_origin=img_origin,
                                             img_tensor=img_tensor,
                                             out=out)
    enc_out_img = encode_image(out_img)
    cls = out_bbox_info.cls.detach().cpu().numpy()
    conf = out_bbox_info.conf.detach().cpu().numpy()
    coord = out_bbox_info.data[:, :4].detach().cpu().numpy()

    res = {'enc_out_img': enc_out_img, 'cls': cls, 'conf': conf, 'coord': coord}
    return res