
import numpy as np
import pandas as pd
import os, gdown, cv2, shutil, math
import pickle
import streamlit as st

import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageOps

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

st.subheader("Siddhesh Masurkar")
#st.text("Masters In Computer Engineering Final Semester Project")

@st.cache_resource

#labels = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def load_dl_model(name):
    model=tf.keras.models.load_model(name)
    return model

def load_ml_model(name):
    #model=tf.keras.models.load_model(name)
    #return model
    pass

model_dictonary = {
            "Random Forest Classifier": 'rf_model.pkl',
            "SVM Classifier": 'svm_model.pkl',
            "Convolution Neural Network": 'cnnkeras.h5',
            "DenseNet121": 'DenseNet121.h5',
            "Xception": 'xception_model.h5' 
        }

options = ['Convolution Neural Network','DenseNet121','Xception','Random Forest Classifier','SVM Classifier']

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):

        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

def get_model_summary(option):
    model = load_dl_model(model_dictonary[sel_option])
    summary = model.summary()
    return summary 


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=224)
    
    sel_option = st.selectbox('Select the Model:',options)
    st.write(f"Selected Model: {model_dictonary[sel_option]}")
    
    button1 = st.button("Prediction")
    if button1:
        st.write("Predict button is pressed..")
        st.subheader("Predicted Output")
        
        if sel_option == 'Convolution Neural Network':
            cnnmodel = load_model('cnnkeras.h5')
            #image = Image.open(image)
            image = ImageOps.fit(image,(224,224),Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32)/127.5)-1
            data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)
            data[0] = normalized_image_array
            prediction = cnnmodel.predict(data)
            index = np.argmax(prediction)
            #print(index)
            class_name = ['glaucoma', 'cataract', 'normal', 'diabetic_retinopathy']
            class_name = class_name[index]
            confidence_score = prediction[0][index]
            st.success(f'The image most likely belongs to {class_name} class.')
            st.success('Confidence Score is {:.2f} %'.format(confidence_score*100))

        if sel_option == 'DenseNet121':
            densemodel = load_model('DenseNet121.h5')
            #image = Image.open(image)
            image = ImageOps.fit(image,(300,300),Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32)/255)
            data = np.ndarray(shape=(1,300,300,3),dtype=np.float32)
            data[0] = normalized_image_array
            prediction = densemodel.predict(data)
            index = np.argmax(prediction)
            #print(index)
            class_name = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
            #class_name = ['glaucoma', 'cataract', 'normal', 'diabetic_retinopathy']
            class_name = class_name[index]
            confidence_score = prediction[0][index]
            st.success(f'The image most likely belongs to {class_name} class.')
            st.success('Confidence Score is {:.2f} %'.format(confidence_score*100))

        if sel_option == 'Xception':
            xceptionmodel = load_model('xception_model.h5')
            #image = Image.open(image)
            image = ImageOps.fit(image,(224,224),Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32)/255)
            data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)
            data[0] = normalized_image_array
            prediction = xceptionmodel.predict(data)
            index = np.argmax(prediction)
            #print(index)
            class_name = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
            #class_name = ['normal', 'glaucoma', 'diabetic_retinopathy', 'cataract']
            class_name = class_name[index]
            confidence_score = prediction[0][index]
            st.success(f'The image most likely belongs to {class_name} class.')
            st.success('Confidence Score is {:.2f} %'.format(confidence_score*100))
        
        if sel_option == 'Random Forest Classifier':
            rf_model = pickle.load(open('rf_model.pkl','rb'))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 128))  # Adjust the size as needed
            image = image / 255.0  # Normalize pixel values
            image= image.reshape(1, 16384)
            class_name = ["glaucoma", "cataract", "normal", "diabetic_retinopathy"]
            pred = rf_model.predict(image)
            proba = rf_model.predict_proba(image)
            #print(pred)
            #prediction = class_name[np.array(pred)]
            prediction = np.array(class_name)[pred.astype(int)]
            confidence = proba[0][pred[0]]
            st.success(f'The image most likely belongs to {prediction[0]} class.')
            st.success('Confidence Score is {:.2f} %'.format(confidence*100))

        if sel_option == 'SVM Classifier':
            model = pickle.load(open('svm_model.pkl','rb'))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 128))  # Adjust the size as needed
            image = image / 255.0  # Normalize pixel values
            image= image.reshape(1, 16384)
            class_name = ["glaucoma", "cataract", "normal", "diabetic_retinopathy"]
            pred = model.predict(image)
            proba = model.predict_proba(image)
            #print(pred)
            #prediction = class_name[np.array(pred)]
            prediction = pred[0]
            index = np.argmax(proba)
            confidence = proba[0][index]
            st.success(f'The image most likely belongs to {prediction} class.')
            st.success('Confidence Score is {:.2f} %'.format(confidence*100))

