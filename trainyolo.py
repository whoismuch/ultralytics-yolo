from ultralytics import YOLO
import sys
import os
import argparse


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('--epochs', default='3')
    parser.add_argument ('--weight_decay', default='0.0005')
    parser.add_argument ('--dfl',  default='1.5')
    parser.add_argument ('--cls',  default='0.5')
    parser.add_argument ('--box',  default='7.5')
    parser.add_argument ('--pose',  default='12.0')
    parser.add_argument ('--kobj',  default='2.0')
    parser.add_argument ('--translate',  default='2.0')
    parser.add_argument ('--degrees',  default='2.0')
    parser.add_argument ('--shear',  default='2.0')


    return parser


def train(epochs, weight_decay, dfl, cls, box, pose, kobj, translate, degrees, shear):
    model = YOLO("yolov5s.yaml").load("yolov5s.pt")
    # Train the model
    results = model.train(data="./datasets/coco128.yaml", epochs=int(epochs), imgsz=640, seed=0,
                          weight_decay=float(weight_decay),
                          dfl=float(dfl), cls=float(cls), box=float(box), pose=float(pose), kobj=float(kobj),
                          translate=float(translate), degrees=float(degrees), shear=float(shear))
    return results


if __name__ == "__main__":
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    train(namespace.epochs,
          namespace.weight_decay, namespace.dfl,
          namespace.cls, namespace.box,
          namespace.pose, namespace.kobj,
          namespace.translate, namespace.degrees, namespace.shear)