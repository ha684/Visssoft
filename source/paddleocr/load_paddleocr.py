from paddleocr import PaddleOCR
def load_paddleocr():
    model = PaddleOCR(help='==SUPPRESS==',use_angle_cls=False,lang='vi',show_log = False)
    return model


