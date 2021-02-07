# TEAM ENIGMA
import os
import numpy as np
from cv2 import *
import tensorflow as tf
from Generator import *
from glob import glob as files

# Size of the Image
size = 800

# Other Params
sizeBlank, img_no, isDrawn, stroke_size = 20, 0, False, 3
font = FONT_HERSHEY_SIMPLEX
_x, _y = -1, -1
name = "Object Remover"
WINDOW_SIZE = (1280, 720)
RES_STORAGE = 'Result'

# Prerained model path
pretrained_model = './Model/pre_model'

# Paths for image files strored in folder testimages
path_images = []
path_images.extend(sorted(files(os.path.join('Data/', '*.jpg'))))

# Extract Image from given path and pre-process it
def get_image(num = 0):
    global path_images, img_no
    if num == 1:
        img_no -= 1
    if num == -1:
        img_no -= 2
        if img_no < 0:
            img_no = len(path_images) - 1
    else:
        if img_no >= len(path_images):
            img_no = 0

    # print(f"Image Number {img_no}")
    image = imread(path_images[img_no])
    image = resize(image, (size, size))
    image = image / 255.0
    img_no += 1
    return image

# Mask Generation
def masking(image):
    mask = (np.array(image[:, :, 0]) == 0.9)
    mask = mask & (np.array(image[:, :, 1]) == 0.9)
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

    # Window Setup
    namedWindow(name, WINDOW_NORMAL)
    resizeWindow(name, WINDOW_SIZE[0], WINDOW_SIZE[1])
    setMouseCallback(name, call_mouse)
    
    # Tensorflow and Model Init
    sess = tf.compat.v1.InteractiveSession()
    isTr = tf.compat.v1.placeholder(tf.bool)
    images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[1, size, size, 3], name="images")

    # Initiliasing our Constructor Class
    model = Generate()

    # Restoring save points for faster performnace
    recon_gen = model.generator(images_placeholder, isTr)
    model_saver = tf.compat.v1.train.Saver(max_to_keep=100)
    model_saver.restore(sess, pretrained_model)

    # Trackbar for this Project
    createTrackbar('Pen Size', name, 8, 25, lambda x: x/2)

    # Image Placeholders Setup
    image = get_image()
    text_box = np.zeros((sizeBlank, 2*size + sizeBlank, 3)) + 1.
    empty = np.zeros((size, size, 3))
    blank = np.zeros((size, sizeBlank, 3)) + 1
    
    # Set generated image to Empty
    gen_img = empty
    
    # Main Loop
    while(True):
        # Windows and text
        sub_window = np.hstack((image, blank, gen_img[:, :, [2, 1, 0]]))
        window = np.vstack((sub_window, text_box))
        imshow(name, window)
        putText(text_box, 'Image', (350, 15), font, 0.7, (0, 255, 0), 2, LINE_AA)
        putText(text_box, 'Reconstructed Image', (300 + size, 15), font, 0.7, (255, 0, 0), 2, LINE_AA)

        # Key Events
        key = waitKeyEx(1)
        # if key != -1:
        #     print(key)
        
        # ESC Key
        if key == 27:
            break

        # Enter Key or F key for Results
        elif key == 13 or key == 102:
            input_masked = masking(image)
            input_masked = input_masked[:, :, [2, 1, 0]]
            shape = np.array(input_masked).shape
            input_tensor = np.array(input_masked).reshape(1, shape[0], shape[1], shape[2])
            output_tensor = sess.run(
                recon_gen,
                feed_dict={
                    images_placeholder: input_tensor,
                    isTr: False
                }
            )
            gen_img = np.array(output_tensor)[0, :, :, :].astype(float)
            # imwrite(os.path.join('inputs', path_images[img_no][21 : 35]), ((image) * 255))

        # S Key for saving the Results
        elif key == 115:
            res_path = RES_STORAGE + f'/res_{img_no}.jpg'
            imwrite(res_path, ((gen_img[:,:,[2, 1, 0]]) * 255))
            print(f"Result saved at {res_path}")

        # R key to reset the current progress
        elif key == 114:
            image = get_image(1)
            gen_img = empty

        # D key for next image
        elif key == 100:
            image = get_image()
            gen_img = empty

        # A key for previous image
        elif key == 97:
            image = get_image(-1)
            gen_img = empty
        
        # Support for closing using Alt+F4 and Cross Button on Window
        if getWindowProperty(name, WND_PROP_VISIBLE) < 1:        
            break

        # Adjust pen size
        stroke_size = getTrackbarPos('Pen Size', name)

    destroyAllWindows()
    # TEAM ENIGMA
