# src/save_parameters.py
import os
import torch

def save_model_parameters(model, folder_name='model_parameters'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    torch.save(model.state_dict(), os.path.join(folder_name, 'trained_model_parameters.pth'))
    print(f"Model parameters saved in '{folder_name}'.")
