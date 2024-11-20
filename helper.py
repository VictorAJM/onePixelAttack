import pickle
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from matplotlib import pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm

def perturb_image(xs, img):
  if xs.nmdim < 2:
    xs = np.array([xs])
  
  tile = [len(xs)] + [1]*(xs.dim+1)
  imgs = np.tile(img, tile)
  xs = xs.astype(int)
  
  for x, img in zip(xs, imgs):
    pixels = np.split(x, len(x) // 5)
    for pixel in pixels:
      x_pos, y_pos, *rgb = pixel
      img[x_pos, y_pos] = rgb
  return imgs

def