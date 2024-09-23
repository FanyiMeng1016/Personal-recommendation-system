import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
import numpy as np
import os
import pickle

features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

# ResNet50 and GlobalMaxPooling2D
weights_path = '/Users/mengfy/Desktop/coding/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = ResNet50(weights=weights_path, include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# set title
st.title('Personalized Style Recommender System')

def save_file(uploaded_file):
    try:
        upload_path = os.path.join("uploader", uploaded_file.name)
        with open(upload_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return upload_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    result = result_to_resnet.flatten()
    normlized_result = result / norm(result)
    return normlized_result

def recommend(features, features_list):
    
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

def update_user_preferences(uploaded_img_path):
    # update preferences
    user_img_features = extract_features(uploaded_img_path, model)
    # save
    with open("user_preferences.pkl", "ab") as f:
        pickle.dump(user_img_features, f)

uploaded_file = st.file_uploader("Upload your image")
if uploaded_file is not None:
    # save picture
    saved_path = save_file(uploaded_file)
    if saved_path:
        # show images
        show_images = Image.open(saved_path)
        size = (400, 400)
        resized_im = show_images.resize(size)
        st.image(resized_im)
        
        # update
        update_user_preferences(saved_path)
        
        user_features = extract_features(saved_path, model)
        img_indices = recommend(user_features, features_list)
        
        # show recommendation
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(img_files_list[img_indices[0][0]])
        with col2:
            st.image(img_files_list[img_indices[0][1]])
        with col3:
            st.image(img_files_list[img_indices[0][2]])
        with col4:
            st.image(img_files_list[img_indices[0][3]])
        with col5:
            st.image(img_files_list[img_indices[0][4]])
    else:
        st.error("Error occurred while uploading the file.")
