# Introduction
**Image Inpainting** is a classical problem in computer vision and graphics.<br><br>
The objective is to fill semantic and reasonable contents in the corruptions and voids to make the completre image. Humans can fill the missing regions by the empirical knowledge to the diverse
object structures from the real world.<br><br>
Nevertheless, it is not easy for machines to learn a
wide variety of structures in natural images and predict what to fill in unknown missing-data
regions in images. Thus, it is crucial to know how to learn an image transformation from a
corrupted image with missing data to a completed image.<br><br>
The current prototype is based on the following Research Paper: [Stanford Paper Link](http://stanford.edu/class/ee367/Winter2018/fu_guan_yang_ee367_win18_report.pdf)<br>
[Link to DataSet](https://content.alegion.com/datasets/coco-ms-coco-dataset)

# Object Removal using cGANs
This project tries to achieve object removal from images and get the base image reconstructed based on surrounding pixels(objects and colours) using conditional **Generative Adversarial Networks** (cGANs).

### 🛠 &nbsp;Tech Stack
![Tensorflow](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white)&nbsp;
![Python](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)&nbsp;
![Numpy](https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white)&nbsp;
<br>

### 🔭 Snapshots
<img src="Data/Readme/img1.png">
Working of this Project
<br><br>
<img src="Data/Readme/img2.png">
Code Example
<br><br>

### How to Use:
Step 1: Install all the required dependencies.
<br><br>
Step 2: Open the **Main.py** file on your code editor and run it.
<br><br>
Step 3: Use the following Hot Keys to navigate through the UI:
<br>
 **[A] Key**          : Moves to next Image.
 <br>
 **[D] Key**          : Moves to previous Image.
 <br>
 **[Esc] Key**        : Close the Windowed Application.
 <br><br>
Step 4: Use the bottom bar to increase or decrease the pen size and select the part of image to remove by dragging the mouse over them(just like painting).
<br><br>
Step 5: Then use the following hot keys as per the requirement: 
<br>
 **[F] or[Enter] Key**: Applies the filter, removes the selected the elements and displays the resulting image on right panel.
 <br>
 **[R] Key**          : Resets the masked Image to Original(undo all the masking).
 <br>
 **[S] Key**          : Save the resulting image.
<br>
