<h2 align="center"> Image Classification for Pneumonia Detection Using Chest X-Ray Images</h2>
<div align="right">
<img src="Image Folder/USD Logo.png" alt="Logo" width="80" height="80">
</div>
<h4 align="center"> ADS 599 Capstone Project <br /> 
  Applied Data Science Master’s Program <br /> 
  Shiley Marcos School of Engineering / University of San Diego 
</h4>

## Description
With the medical resource’s shortage, we are trying to look for ways to make healthcare more efficient. Our group aims to use machine learning and deep learning techniques with image classification segments to deploy a product that can identify chest X-ray scans as either an image presenting with pneumonia or without pneumonia with a probability as a confidence score.

## Dependencies
We developed the data preparation pipeline and multiple machine learning and deep learning models using TensorFlow and scikit-learn Python libraries in `jupyter notebook`. This final model would then be saved and deployed via Flask in `Spyder`. 
```sh
import tensorflow
import sklearn
import flask
```

## Dataset Description:
Dataset Name: Chest X-Ray Images (Pneumonia)

Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Our dataset consists of 5,856 X-ray images that split into train, validate and test sets. Each set is categorized into "NORMAL" and "PNEUMONIA". For the training set, we have 1341 "NORMAL" images and 3875 "PNEUMONIA" images; For the validate set, we have 8 images each for "NORMAL" AND "PNEUMONIA"; And for the test set, we have 234 images for "NORMAL" and 390 images for "PNEUMONIA" which sums up at 5856 images total.

## Features
Flask deployment will be initial an HTML file. From the web page, the user can feed an image for classification by loading a selected image via the file selection option. The user will then prompt the application to begin the classification pipeline by submitting the image via the “Classify” button.

 <img src="Image Folder/Flask Demo.png" alt="Logo" width="480" height="400">

## Author
Contributors names and contact info

* <a href="https://github.com/dingyiduan7"> Dingyi Duan </a> <br />
* <a href="https://github.com/evchow"> Eva Chow </a> <br />
* <a href="https://github.com/Abby-Tan"> Abby Tan </a> <br />

## Version History
* 0.1 Initial Release
