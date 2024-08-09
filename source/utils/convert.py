from pdf2image import convert_from_bytes
import os

poppler_path = 'Poppler/poppler-24.02.0/Library/bin'

if not os.path.exists(poppler_path):
    poppler_link = 'https://poppler.freedesktop.org/'
    raise Exception(f'Poppler not found. Please install it from: {poppler_link}')

def auto_scroll():
    js = """
    <script>
        var body = window.parent.document.querySelector(".main");
        body.scrollTop = body.scrollHeight;
    </script>
    """

def pdf_to_images(pdf_path):
    images = convert_from_bytes(pdf_path, poppler_path=poppler_path)
    return images

