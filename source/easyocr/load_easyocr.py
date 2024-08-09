import easyocr

def load_easyocr():
    return easyocr.Reader(['vi'],download_enabled=False,quantize=False)
