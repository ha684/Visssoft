import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
import torch
from craft import CRAFT
from craft_predict import copyStateDict
from refinenet import RefineNet

# ============================================================
trained_model = './weights/craft_mlt_25k.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
refiner_model = './weights/craft_refiner_CTW1500.pth'
# ============================================================
def load_model_craft():
    
    net = CRAFT().to(device)
    refine_net = RefineNet().to(device)
    net.load_state_dict(copyStateDict(torch.load(trained_model,map_location=device,weights_only=True)))
    refine_net.load_state_dict(copyStateDict(torch.load(refiner_model,map_location=device,weights_only=True)))
    net = torch.nn.DataParallel(net)
    refine_net = torch.nn.DataParallel(refine_net)
    
    return net.eval(),refine_net.eval()
