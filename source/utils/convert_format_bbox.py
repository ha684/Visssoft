import numpy as np

def convert_poly_to_rectangle(list_polygon):
    top_left, top_right, bottom_left, bottom_right = list_polygon
    x_coords = [top_left[0], top_right[0], bottom_right[0], bottom_left[0]]
    y_coords = [top_left[1], top_right[1], bottom_right[1], bottom_left[1]]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    return list(map(int, [min_x, min_y, max_x, max_y]))

def xyxy_to_corners(bbox):
    x_min, y_min, x_max, y_max = bbox
    top_left = [x_min, y_min]
    top_right = [x_max, y_min]
    bottom_right = [x_max, y_max]
    bottom_left = [x_min, y_max]
    return [top_left, top_right, bottom_right, bottom_left]

def convert_xyxy2xywh(list_coor):
    x = list_coor[0]
    y = list_coor[1]
    w = list_coor[2] - list_coor[0]
    h = list_coor[3] - list_coor[1]
    return [x, y, w, h]

def convert_xywh2xyxy(list_coor):
    x1 = list_coor[0]
    y1 = list_coor[1]
    x2 = list_coor[0] + list_coor[2]
    y2 = list_coor[1] + list_coor[3]
    return [x1, y1, x2, y2]

def get_cropped_area(image_arr, bbox):
    return image_arr[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def sort_boxes(boxes, max_y_diff=10, yolo=False):
    if yolo:
        def box_key(box):
            # Get the top-left corner of the box
            x, y = box[0], box[1]
            # Calculate the row number (allowing for some vertical overlap)
            row = y // max_y_diff
            # Return a tuple for sorting: (row, x-coordinate)
            return (row, x)
    else:
        def box_key(box):
            # Get the top-left corner of the box
            x, y = box[0]
            # Calculate the row number (allowing for some vertical overlap)
            row = y // max_y_diff
            # Return a tuple for sorting: (row, x-coordinate)
            return (row, x)

    return np.array(sorted(boxes, key=box_key)).astype(int)

def process_boxes(boxes):
    boxes = sort_boxes(boxes)
    boxes = [convert_poly_to_rectangle(box) for box in boxes]
    return boxes