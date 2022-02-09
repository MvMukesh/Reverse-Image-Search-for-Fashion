import streamlit as st
import os
from PIL import Image
import numpy as np
from numpy.linalg import norm
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

feature_list = pickle.load(open('embedding.pkl', 'rb')) #image features
feature_list = np.array(feature_list)
filenames = pickle.load(open('filenames.pkl', 'rb'))  #image file name

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #will use our own top layer
model.trainable = False #no need to train our model
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()]) #adding our layer
#print(model.summary())

st.title('Reverse Image Search for Fashion')

"""
Steps:
1. Save user uploaded image to uploaded_image file
2. Load file
3. Extract Features
4. Show similar images using Recommendation
"""
### 1 ###
def save_uploadedfile(uploaded_file):
    try:
        with open(os.path.join('uploaded_image',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
### 2 ###
def feature_extraction(img_path,model):
    img = image.load_img('sample/half.jpeg', target_size=(224, 224))  # loading image
    img_array = image.img_to_array(img)  # image to array
    expanded_img_array = np.expand_dims(img_array, axis=0)  # to batch
    preprocessed_img = preprocess_input(expanded_img_array)
    output = model.predict(preprocessed_img).flatten()
    output_normalized = output / norm(output)
    return output_normalized
### 3 ###
def recommend(features,features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

#1
uploaded_file = st.file_uploader('Put one image here')
if uploaded_file is not None:
    if save_uploadedfile(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #2
        features = feature_extraction(os.path.join('uploaded_image', uploaded_file.name), model)
        #3
        indices = recommend(features,feature_list)
        #4 show
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header('Error in file upload')

