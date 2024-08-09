from collections import OrderedDict
from source.craft import craft_utils, imgproc
import cv2
import torch
from torch.autograd import Variable
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# ============================================================
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4 
cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
canvas_size = 1280
mag_ratio = 1.5
poly = False

# ============================================================

def test_net(net, image, text_threshold,link_threshold,low_text,cuda,poly,refine_net = None):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image,canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))    # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.to(cuda)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()
    #post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold,link_threshold,low_text,poly)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
        
    return polys
def predict_craft(net, refine_net,image_path):
    # Load the image
    image = imgproc.loadImage(image_path)
    polys = test_net(
        net, 
        image, 
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        poly=poly,
        cuda=cuda,
        refine_net = refine_net
    )
    polys = [poly.tolist() for poly in polys]
    
    return polys

