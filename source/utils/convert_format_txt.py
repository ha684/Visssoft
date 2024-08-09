from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from collections import defaultdict
font_path = './font/DejaVuSansCondensed.ttf'

def create_text_boxes(boxes, texts):
    if len(boxes) == 0:
        return [{'box': [0,0,0,0], 'text': '###'}]
    text_boxes = [{'box': box, 'text': text} for box, text in zip(boxes, texts)]
    return text_boxes

def create_pdf(text_boxes, output_file,font_path):
    c = canvas.Canvas(output_file, pagesize=letter)
    pdfmetrics.registerFont(TTFont('DejaVu',font_path))
    c.setFont('DejaVu', 12)
    for item in text_boxes:
        x1, y1, x2, y2 = item['box']
        c.drawString(x1, letter[1] - y1, item['text'])
    c.save()

def create_line_boxes(text_boxes):
    text_boxes_copy = text_boxes[:]
    hash = defaultdict(list)
    count = 1
    while text_boxes_copy:
        cur_box = text_boxes_copy[0]['box']
        cur_y = cur_box[1]
        hash[count].append(cur_box)
        text_boxes_copy.pop(0)
        if text_boxes_copy:
            next_y = text_boxes_copy[0]['box'][1]
            if abs(next_y - cur_y) < 10:
                continue
        count += 1
    
    return hash

def create_line_txts(text_boxes):
    root = min(box['box'][0] for box in text_boxes)
    text_boxes_copy = text_boxes[:]
    line_boxes = create_line_boxes(text_boxes_copy)
    line_txts = defaultdict(list)
    for k,v in line_boxes.items():
        v_sorted = sorted(v, key=lambda box: box[0]) 
        for box in v_sorted:
            for item in text_boxes:
                if item['box'] == box:
                    if box[0] > root:
                        distance = box[0] - root
                        tabs = distance // 35
                        line_txts[k].append('\t' * tabs + item['text'])
                    else:
                        line_txts[k].append(item['text'])
                    break
    return line_txts

def correct_format(text_boxes):
    corrected_format = []
    line_boxes = create_line_boxes(text_boxes)
    line_txts = create_line_txts(text_boxes)
    previous_y = None
    for k in sorted(line_txts.keys()):
        current_y = None
        for item in text_boxes:
            if item['box'] in line_boxes[k]:
                current_y = item['box'][1]
                break
        if previous_y is not None and current_y is not None:
            distance = current_y - previous_y
            if distance > 20: 
                extra_newlines = distance // 25
                corrected_format.append('\n' * extra_newlines)
        
        corrected_format.append(f"{''.join(line_txts[k])}\n")
        previous_y = current_y
    return corrected_format
