def predict_vietocr(detector, images):
    pred, prob = detector.predict_batch(images , return_prob = True)
    return [pred, prob]