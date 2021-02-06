from cv2 import *
import numpy as np

# Function to generate mask for input as per the stroke
def masking(image):
    mask = (np.array(image[:, :, 0]) == 0.9)
    mask = mask & (np.array(image[:, :, 1]) == 0.9)
    mask = mask & (np.array(image[:, :, 2]) == 0.9)
    mask = np.dstack([mask, mask, mask])
    return (True ^ mask) * np.array(image)


# Function for drawing lines on objects to be removed
def mouse_callback(mouse_event, x, y, flags, parameters):
    global _x, _y, isDrawn
    if mouse_event == EVENT_LBUTTONDOWN:
        isDrawn = True
        _x, _y = x, y

    elif mouse_event == EVENT_MOUSEMOVE:
        if isDrawn:
            line(image, (_x, _y), (x, y), (0.0, 0.0, 0.0), stroke_size)
            _x, _y = x, y

    elif mouse_event == EVENT_LBUTTONUP:
        isDrawn = False
        line(image, (_x, _y), (x, y), (0.0, 0.0, 0.0), stroke_size)