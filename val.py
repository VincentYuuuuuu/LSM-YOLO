import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('Weights Here')
    model.val(data='Data Yaml Path',
                split='val',
                save_json=False, # if you need to cal coco metrics
                project='runs/val',
                name='exp',
                device='',
                )
