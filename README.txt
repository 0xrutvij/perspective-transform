/* Filename:  README.txt
 * Date:      05/07/2022
 * Author:    Rutvij Shah
 * Email:     rutvij.shah@utdallas.edu
 * Course:    CS6384.001 Spring 2022
 * Version:   1.0
 * Copyright: 2022, All Rights Reserved
 *
 * Description:
 *     Check perspective transform project
 */

------------------------------------------------------------------------

To run the project and obtain b/w output run the following command

  python proj3.py -i sample_input -o sample_bw_output -g

Example output can be found in sample_bw_output folder.

------------------------------------------------------------------------

To run the project and obtain color output run the following command

  python proj3.py --input_folder sample_input --save_to_folder sample_color_output

Example output can be found in sample_color_output folder.

------------------------------------------------------------------------
/model and /src contain the code for Handwriting OCR and are taken from
https://github.com/githubharald/SimpleHTR.git

The /model folder contains are pre-trained model, and src contains the TF
source code to run inference.

To run the project with check amount detection, use the following command

  python proj3.py -i sample_input -o color_output -d 2> /dev/null

Example output can be found in sample_amt_detection_output folder.

------------------------------------------------------------------------

Project requirements are listed in requirements.txt

------------------------------------------------------------------------
.
|-- README.txt
|-- model
|   |-- charList.txt
|   |-- checkpoint
|   |-- snapshot-13.data-00000-of-00001
|   |-- snapshot-13.index
|   |-- snapshot-13.meta
|   `-- summary.json
|-- proj3.py
|-- requirements.txt
|-- sample_amt_detection_output
|   |-- check1_cd_amount.jpg
|   |-- check2_cd_amount.jpg
|   `-- check3_cd_amount.jpg
|-- sample_bw_output
|   |-- check1.jpg
|   |-- check2.jpg
|   |-- check3.jpg
|   `-- check5.jpg
|-- sample_color_output
|   |-- check1.jpg
|   |-- check2.jpg
|   |-- check3.jpg
|   `-- check5.jpg
|-- sample_input
|   |-- check1.jpg
|   |-- check2.jpg
|   |-- check3.jpg
|   `-- check5.jpeg
`-- src
    |-- __init__.py
    |-- create_lmdb.py
    |-- dataloader_iam.py
    |-- main.py
    |-- model.py
    `-- preprocessor.py

6 directories, 30 files



------------------------------------------------------------------------
------------------------------------------------------------------------

Approach

Check perspective transform

(a) Approach 1
  (1) Threshold, Gaussian blur and detect edges in the image (canny edge detection).
  (2) Extract the contours in the image
  (3) Isolate the largest contour by area
  (4) Use the bounding-box defined by the contour to define the perspective transform parameters
  (5) Display the transformed image (rotate if the image is not in landscape orientation)

(b) Approach 2
  (1) If approach 1 fails to create an image
  (2) Isolate the axis parallel edges (vertical and horizontal)
  (3) Overlay the edges and find the leftmost, rightmost, topmost and bottom most edges
  (4) Use the intersection of those 4 edges to determine the bounding box.
  (5) Use the bounding-box defined by the contour to define the perspective transform parameters
  (6) Display the transformed image (rotate if the image is not in landscape orientation)


Check amount detection

(1) Crop the transformed check image to include only the last third and top 2/3rd of the cheque
(2) Apply canny edge detection after gaussian smoothing and find contours
(3) Extract the largest possible quadrilateral contour (4 points exactly)
(4) Crop image to that the size of the contour and extract the text in amount box
(5) Use an handwriting text OCR model to detect the amount
(6) Overlay the detected amount and box on the image and display it.
