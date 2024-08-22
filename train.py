import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/LSM-YOLO/LSM-YOLO.yaml')
    model.train(data='Data Yaml Path',
                cache=False,
                project='runs/train',
                name='exp',
                epochs=300,
                batch=48,
                close_mosaic=0,
                optimizer='SGD', # using SGD
                device='',
                # resume='', # last.pt path
                )