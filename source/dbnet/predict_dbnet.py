from source.dbnet.utils import test_preprocess, read_img,return_bbox
import torch 
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
def predict_dbnet(model,image_arr):
    img, h_origin, w_origin = read_img(image_arr)
    preprocessed_image = test_preprocess(img,to_tensor=True).to(device)
    with torch.no_grad(): preds = model(preprocessed_image)
    batch = {'shape': [(h_origin, w_origin)]}
    boxes,scores = return_bbox(batch,preds)
    return boxes,scores