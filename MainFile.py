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
    print("Window Setup")
    image = get_image()
    text_box = np.zeros((sizeBlank, 2*size + sizeBlank, 3)) + 1.
    empty = np.zeros((size, size, 3))
    blank = np.zeros((size, sizeBlank, 3)) + 1

    namedWindow("Object Removal Window", WINDOW_NORMAL)
    setMouseCallback('Object Removal Window', call_mouse)
    
    # Prerained model path
    pretrained_model = './Model/pre_model'

    # Tensorflow and Model Init
    sess = tf.compat.v1.InteractiveSession()
    isTraining = tf.compat.v1.placeholder(tf.bool)
    images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[1, size, size, 3], name="images")

    # Initiliasing our remover
    model = Reconstruct()

    recon_gen = model.generator(images_placeholder, isTraining)
    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    saver.restore(sess, pretrained_model)

    createTrackbar('Pen Size', 'Object Removal Window', 1, 10, lambda x: x)
    # Widget for pensize

    filtered_image = empty
    # Filtered image which is initially empty

    while(True):
        # Windows and text
        sub_window = np.hstack((image, blank, filtered_image[:, :, [2, 1, 0]]))
        window = np.vstack((sub_window, text_box))
        imshow('Object Removal Window', window)
        putText(text_box, 'Image', (110, 15), font, 0.4, (0, 0, 0), 1)
        putText(text_box, 'Reconstructed Image', (130 + size, 15), font, 0.4, (0, 0, 0), 1)

        # Key Events
        key_pressed = waitKey(1) & 0xFF

        # ESC Key
        if key_pressed == 27:
            break

        # F Key for filter
        elif key_pressed == 102:
            input_masked = masking(image)
            input_masked = input_masked[:, :, [2, 1, 0]]
            shape = np.array(input_masked).shape
            input_tensor = np.array(input_masked).reshape(
                1, shape[0], shape[1], shape[2])
            output_tensor = sess.run(
                recon_gen,
                feed_dict={
                    images_placeholder: input_tensor,
                    isTraining: False
                }
            )
            filtered_image = np.array(output_tensor)[0, :, :, :].astype(float)
            # imwrite(os.path.join('results', path_images[img_no][21 : 35]), ((filtered_image[:,:,[2, 1, 0]]) * 255) )
            # imwrite(os.path.join('inputs', path_images[img_no][21 : 35]), ((image) * 255))

        # R key to reset
        elif key_pressed == 114:
            image = get_image(this=True)
            filtered_image = empty

        # Adjust pen size
        stroke_size = getTrackbarPos('Pen Size', 'Object Removal Window')

    destroyAllWindows()
