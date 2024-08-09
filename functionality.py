from source.craft.load_model import load_model_craft
from source.craft.craft_predict import predict_craft
from source.vietocr.load_model_vietocr import load_model_vietocr
from source.vietocr.vietocr_predict import predict_vietocr
from PIL import Image
import cv2
import torch
import gc
from source.utils.convert_format_bbox import convert_poly_to_rectangle, sort_boxes, get_cropped_area
# from source.dbnet.utils import draw_bbox


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_all_models():
    models = {}
    models['vietocr'] = load_model_vietocr()
    models['craft'], models['craft1'] = load_model_craft()
    return models
models = load_all_models()

def detect(models, image):
    boxes = predict_craft(models['craft'], models['craft1'], image)
    return sort_boxes(boxes)

def recog(models, boxes, main_image):
    cropped_image_batch = [Image.fromarray(get_cropped_area(main_image, box)) for box in boxes]
    results = predict_vietocr(models['vietocr'],[image for image in cropped_image_batch])
    return results

def process_image(image):
    if image is None:
        return None,None
    boxes = detect(models, image)
    # result_image = draw_bbox(image, boxes)
    boxes = [convert_poly_to_rectangle(box) for box in boxes]
    results = recog(models, boxes, image)
    return (
        results[0], # return text pred
        results[1] # return probability
        )
    # text_boxes = create_text_boxes(boxes, pred_texts)
    # corrected_format_text = correct_format(text_boxes)
    # clear_gpu_memory()
    # return result_image, corrected_format_text

if __name__ == '__main__':
    img_path = "12.jpg"
    
    img = cv2.imread(img_path)
    print(process_image(img))
