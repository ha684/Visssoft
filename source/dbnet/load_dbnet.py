from source.dbnet.models import DBTextModel
import torch
import streamlit as st
weights = './weights/best_dbnet.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_dbnet():
    dbnet = DBTextModel().to(device)
    dbnet.load_state_dict(torch.load(weights, map_location=device,weights_only=True))
    return dbnet.eval()

