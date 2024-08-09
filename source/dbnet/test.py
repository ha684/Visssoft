import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import time
import argparse

import torch

from models import DBTextModel
from utils import (read_img, test_preprocess, visualize_heatmap,
                   visualize_polygon, str_to_bool)


def load_model(args):
    assert os.path.exists(args.model_path)
    dbnet = DBTextModel(plus=False).to(args.device)
    dbnet.load_state_dict(torch.load(args.model_path,
                                     map_location=args.device))
    return dbnet


def load_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--image_path', type=str, default='./assets/foo.jpg')
    parser.add_argument('--model_path',
                        type=str,
                        default='./models/db_resnet18.pth')
    parser.add_argument('--folder_path', type=str, default="")
    parser.add_argument('--save_dir', type=str, default='./assets')
    parser.add_argument('--device', type=str, default='cuda')

    # for heatmap
    parser.add_argument('--prob_thred', type=float, default=0.45)

    # for polygon & rotate rectangle
    parser.add_argument('--heatmap', type=str_to_bool, default=False)
    parser.add_argument('--thresh', type=float, default=0.2)
    parser.add_argument('--box_thresh', type=float, default=0.5)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--is_output_polygon', type=str_to_bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.6)

    args = parser.parse_args()
    return args

def test_image(net, img_path, args):
    img_fn = img_path.split("/")[-1]
    assert os.path.exists(img_path)
    
    img_origin, h_origin, w_origin = read_img(img_path)
    tmp_img = test_preprocess(img_origin, to_tensor=True,
                              pad=False).to(args.device)

    net.eval()
    torch.cuda.empty_cache()
    gc.collect()

    start = time.time()
    with torch.no_grad():
        preds = net(tmp_img)
    print(">>> Inference took {}'s".format(time.time() - start))
    
    if args.heatmap:
        visualize_heatmap(args, img_fn, tmp_img, preds.to('cpu')[0].numpy())
    else:
        batch = {'shape': [(h_origin, w_origin)]}
        visualize_polygon(args, img_fn, (img_origin, h_origin, w_origin),
                          batch, preds)

def main(net, args):
    if args.folder_path != "":
        for image_name in os.listdir(args.folder_path):
            if ".txt" in image_name or ".json" in image_name:
                continue
            
            image_path = os.path.join(args.folder_path, image_name)
            print(image_path)
            
            test_image(net, image_path, args)
    else:
        img_path = args.image_path.replace("file://", "")
        test_image(net, img_path, args)


if __name__ == '__main__':
    args = load_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'
    dbnet = load_model(args)

    main(dbnet, args)
