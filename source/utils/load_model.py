import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from source.craft.craft_predict import predict_craft
from source.yolo.predict_yolo import predict_yolov8
from source.easyocr.predict_easyocr import predict_easyocr
from source.paddleocr.predict_paddleocr import predict_paddleocr
from source.utils.convert_format_bbox import convert_poly_to_rectangle, sort_boxes, get_cropped_area, xyxy_to_corners, convert_xyxy2xywh, process_boxes
from source.dbnet.predict_dbnet import predict_dbnet
from source.craft.load_model import load_model_craft
from source.vietocr.load_model_vietocr import load_model_vietocr
from source.yolo.load_yolo import load_yolov8
from source.dbnet.load_dbnet import load_dbnet
from source.easyocr.load_easyocr import load_easyocr
from source.paddleocr.load_paddleocr import load_paddleocr
# from source.Corrector.load_textcorrection import corrector
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
import gc

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
@st.cache_resource
def load_all_models():
    models = {}
    total_models = 6
    progress_text = "Loading models... Please wait."
    my_bar = st.progress(0, text=progress_text)

    models['vietocr'] = load_model_vietocr()
    my_bar.progress(1/total_models, text=progress_text)
    models['dbnet'] = load_dbnet()
    my_bar.progress(2/total_models, text=progress_text)
    models['craft'],models['craft1'] = load_model_craft()
    my_bar.progress(3/total_models, text=progress_text)
    models['yolov8'] = load_yolov8()
    my_bar.progress(4/total_models, text=progress_text)
    models['paddleocr'] = load_paddleocr()
    my_bar.progress(5/total_models, text=progress_text)
    models['easyocr'] = load_easyocr()
    my_bar.progress(6/total_models, text=progress_text)
    my_bar.empty()

    return models

def detect(_models,image_path,model_detect_name):
    if model_detect_name == 'Craft':
        boxes = predict_craft(_models['craft'],_models['craft1'], image_path=image_path)
    elif model_detect_name == 'Yolov8':
        boxes = [xyxy_to_corners(box) for box in predict_yolov8(_models['yolov8'], image_path)]
    elif model_detect_name == 'EasyOCR':
        boxes = [line[0] for line in predict_easyocr(_models['easyocr'], image_path)]
    elif model_detect_name == 'DBNet':
        boxes, scores = predict_dbnet(_models['dbnet'], image_path)
    elif model_detect_name == 'PaddleOCR':
        result = predict_paddleocr(_models['paddleocr'], image_path)
        boxes = [line[0] for line in result[0]]
        boxes = [np.array(box).reshape((4,2)) for box in boxes]
    elif model_detect_name == 'Best Model':
        result_easyocr = predict_easyocr(_models['easyocr'], image_path)
        boxes_easyocr = [line[0] for line in result_easyocr]
        boxes_db, scores_db = predict_dbnet(_models['dbnet'], image_path)
        boxes_easyocr = process_boxes(boxes_easyocr)
        boxes_db = process_boxes(boxes_db)
        boxes = boxes_easyocr + boxes_db
        scores = [0.8] * len(scores_db) + [0.6] * len(result_easyocr)
        boxes_xywh = [convert_xyxy2xywh(box) for box in boxes]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.4, nms_threshold=0.2)
        if len(idxs) > 0:
            idxs = idxs.flatten()
            boxes = [boxes[i] for i in idxs]
    return sort_boxes(boxes)

def recog(_models,boxes,main_image,model_rec_name):
    if model_rec_name == 'VietOCR':
        cropped_image_batch = [Image.fromarray(get_cropped_area(main_image, box)) for box in boxes]
        pred_texts = _models['vietocr'].predict_batch([image for image in cropped_image_batch])
        # pred_texts = corrector(pred_texts)
    else:
        cropped_image_batch =[get_cropped_area(main_image, convert_poly_to_rectangle(box)) for box in boxes]
        pred_texts = []
        for img in cropped_image_batch:
            pred_texts = predict_easyocr(_models['easyocr'],img)
            text = [line[1] for line in pred_texts]
            pred_texts.append(text[0])
    return pred_texts



