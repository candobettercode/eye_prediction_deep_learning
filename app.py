
import numpy as np
import pandas as pd
import os, gdown, cv2, shutil, math
import pickle
import streamlit as st

import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

from sklearn.model_selection import train_test_split, KFold
from keras.layers import *

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow_addons.metrics import F1Score


st.set_page_config(page_title="Eye Disease Prediction!!!", page_icon=":eye:", layout="centered")
# giving a title
st.title(":eye: Eye Disease Prediction")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

st.subheader("Mrs. Swapna Chavan")
st.text("Masters In Computer Engineering Final Semester Project")

model_dictonary = {
            "Random Forest Classifier": 'rf_model.pkl',
            "SVM Classifier": 'svm_model.pkl',
            "Convolution Neural Network": 'cnnkeras.h5',
            "DenseNet121": 'DenseNet121.h5',
            "Inception": 'xception_model.h5' 
        }

def get_model_summary(model_name):
    pass

def main():

    options = ['Random Forest Classifier','SVM Classifier','Convolution Neural Network','DenseNet121','Inception']
    uploaded_file = st.file_uploader("Upload a file to detect", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=224)
        st.write(type(uploaded_file))

    sel_option = st.selectbox('Select the Model:',options)

    st.write(f"Selected Option: {model_dictonary[sel_option]}")

    if sel_option == 'Random Forest Classifier':
        rf_model = pickle.load(open('rf_model.pkl','rb'))
        image = cv2.imread(uploaded_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128))  # Adjust the size as needed
        image = image / 255.0  # Normalize pixel values
        image= image.reshape(1, 16384)
        prediction = rf_model.predict(image)


    model_summary = get_model_summary(sel_option)
    col1, col2 = st.columns([2,1])

    with col1:
        button1 = st.button("Model Performance")
        if button1:
            st.write("Summary/Performance Selected..")
        st.subheader("Model Performance")
        st.text(model_summary)

    with col2:
        button2 = st.button("Predict")
        if button2:
            st.write("Predict button is pressed..")
        st.subheader("Predicted Output")
        st.text(prediction)

    output_paceholder = st.empty()

if __name__ == '__main__':
    main()
