def predict_paddleocr(model,img_path):
    slice = {'horizontal_stride': 300, 'vertical_stride': 500, 'merge_x_thres': 50, 'merge_y_thres': 35}
    result = model.ocr(img_path, cls=False, slice=slice,bin=True)
    return result