from ultralytics import YOLO
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
def load_yolov8():
    model = YOLO('../weights/text_detection_trained.pt')
    model.to(device)
    return model
