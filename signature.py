import streamlit as st
from tensorflow.keras.models import load_model,Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cv2
import os
import PIL
# from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


st.set_option('deprecation.showfileUploaderEncoding',False)
# @st.cache(allow_output_mutation = True)
def load_model():
    model =tf.keras.models.load_model(r'"C:\Users\vidhi\OneDrive\Desktop\Machine_learning\New folder\my_model3.hdf5"')
    return model
model = load_model()
st.write("""
            #Signature Authentication
            """
        )

file = st.file_uploader("Please upload an signature image" , type = ["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np


def import_and_predict(image_data,model):
    img = cv2.imread(image_data, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = np.array(img).reshape(1, 128, 128, 1) / 255.0

# Predict the class of the signatur
    prediction = model.predict(img)
    return prediction


if file is None:
    st.text("Please upload an image file")
else :
    image = Image.open(file)
    file_name = file.name
    print(image)
    print(f"file_name{file_name}")
    st.image(image,use_column_width =True)
    predictions = import_and_predict(r"C:\Users\vidhi\OneDrive\Desktop\Machine_learning\New folder\combine"+"\\"+file_name,model)
    if predictions < 0.05:
        string = "The signature is real."
    else:
        string = "The signature is forged."
    st.success(string)
