"""
Augmented Reality marker Tracker

ways to use:
1. python exp.py --overlay sticker.png
2. python exp.py --video vdo.mp4 --overlay sticker.png --out out.mp4
3. python exp.py --camera 0 --overlay sticker.png --calib cam.npz

Camera calibration file (.npz) format:
np.savex('cam.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


"""

import argparse
import time 
import numpy as np
import cv2
import os

def load_overlay(path, target_size=None):
    """ load overlay image and return RGBA float32 image in 0..1 range"""
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Overlay image not fount: {path}")
