import torch

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

def load_model_vietocr():
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config['cnn']['pretrained']= True
    # config['weights'] = './weights/vgg_seq2seq.pth'
    detector = Predictor(config)
    return detector