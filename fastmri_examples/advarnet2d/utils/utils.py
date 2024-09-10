import torch
from pathlib import Path
import os
from torchvision.utils import save_image
import cv2
import numpy as np

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def create_dir_and_save_image(
    image: torch.tensor,
    save_dir : str,
    save_name : str,
):
    save_dir = Path(save_dir)
    createDirectory(save_dir)
    
    save_image(image, save_dir / Path(save_name))
    
    #cv2.imwrite(str(save_dir / Path(save_name)), image_copy)
    
    return

def create_dir_and_save_npy(
    array: np.array, # 2D image (numpy array)
    save_dir : str,
    save_name : str,
):
    save_dir = Path(save_dir)
    createDirectory(save_dir)
    
    np.save(str(save_dir / Path(save_name)), array)
    
    #cv2.imwrite(str(save_dir / Path(save_name)), image_copy)
    
    return