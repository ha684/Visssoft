def predict_easyocr(model,img,choice = None):
    result = model.readtext(img)
    if choice == 'text':
        texts = [line[1] for line in result]
        return texts
    if choice == 'box':
        boxes = [line[0] for line in result]
        return boxes
    if choice == 'prob':
        probs = [line[2] for line in result]
        return probs
    return result