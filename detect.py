import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('Weights Here')
    model.predict(source='Images Here',
                project='runs/detect',
                name='exp',
                save=True,
                # visualize=True # visualize model features maps
                )