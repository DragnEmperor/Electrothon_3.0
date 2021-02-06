import os
import numpy as np
from cv2 import *
import tensorflow as tf
from Model.Remover_model import Reconstruct
from glob import glob as files

_x, _y = -1, -1

# Size of the Image
size = 800

# Other Params
sizeBlank, img_no, isDrawn, stroke_size = 20, 0, False, 3
font = FONT_ITALIC

# Paths for image files strored in folder testimages
path_images = []
path_images.extend(sorted(files(os.path.join('Data/', '*.jpg'))))

# Extract Image from given path and pre-process it
def get_image(this = False):
    global path_images, img_no
    if this:
        img_no -= 1
    image = imread(path_images[img_no])
    image = resize(image, (size, size))
    image = image / 255.0
    img_no += 1
    if img_no >= len(path_images):
        img_no = 0

    return image

# Mask Generation
def masking(image):
    mask = (np.array(image[:, :, 0]) == 0.9)
    mask = mask & (np.array(image[:, :, 1]) == 0.1)
    mask = mask & (np.array(image[:, :, 2]) == 0.9)
    mask = np.dstack([mask, mask, mask])
    return (True ^ mask) * np.array(image)


# Mouse Callback
def call_mouse(event, x, y, flag, param):
    global _x, _y, isDrawn
    if event == EVENT_LBUTTONDOWN:
        isDrawn = True
        _x, _y = x, y

    elif event == EVENT_MOUSEMOVE:
        if isDrawn:
            line(image, (_x, _y), (x, y), (0.0, 0.0, 0.0), stroke_size)
            _x, _y = x, y

    elif event == EVENT_LBUTTONUP:
        isDrawn = False
        line(image, (_x, _y), (x, y), (0.0, 0.0, 0.0), stroke_size)

# Main Code Starts Here
if __name__ == "__main__":
    print("ToDo")
