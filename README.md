# Introduction - Project Efface
**Image Inpainting** is a classical problem in computer vision and graphics.<br><br>
The objective is to fill semantic and reasonable contents in the corruptions and voids and generate an image. Humans can fill the missing regions using empirical knowledge of the diverse
object structures present in real world.<br><br>
Nevertheless, it is not easy for machines to learn a
wide variety of structures in natural images and predict what to fill in unknown missing-data
regions in images. Thus, it is crucial to know how to learn an image transformation from a
corrupted image with missing data to a completed image.<br><br>
The current prototype is based on the following Research Paper:<br> 
- [Stanford Paper Link](http://stanford.edu/class/ee367/Winter2018/fu_guan_yang_ee367_win18_report.pdf)
- [Link to Dataset](https://content.alegion.com/datasets/coco-ms-coco-dataset)

### 🛠 &nbsp;Tech Stack
![Tensorflow](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white)&nbsp;
![Python](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)&nbsp;
![Numpy](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white)&nbsp;
<br>


# Object Removal using cGANs
This project tries to achieve object removal from images and get the base image reconstructed using surrounding pixels and pretrained model using conditional **Generative Adversarial Networks** (cGANs).

## 🔭 Working Protoytpe Snapshots
- **Working of this Project**: The black regions on the left side represent the objects/elements that we wish to remove.
    
    <img src="Data/Readme/img1.png">
<br>

- **Code Example**

    <img src="Data/Readme/img2.png">
<br>

### How to Use:
- Step 1: Install all the required dependencies. Use the following command to install the dependencies.
    > pip3 install -r requirements.txt

- Step 2: Open the **MainFile.py** file and execute it.

- Step 3: Use the following Hot Keys to navigate through the UI:

    - **[A] Key** : Move to next Image.
 
    - **[D] Key** : Move to previous Image.
 
    - **[Esc] Key** : Close the Windowed Application.
 
- Step 4: Use the slider present at the bottom to increase or decrease the pen size. Then select the part of image you wish to remove by dragging the mouse over them, just like a brush!

- Step 5: Then use the following hot keys as per the requirement: 

    - **[F] or [Enter] Key** : Starts the reconstruction process, removes the selected elements and displays the resulting image on right panel.

    - **[R] Key** : Resets the masked Image to Original(undo all the masking).
 
    - **[S] Key** : Save the resulting image in Result directory.

