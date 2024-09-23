from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import os
from tqdm import tqdm
import pickle
import numpy as np
from numpy.linalg import norm

weights_path = '/Users/mengfy/Desktop/coding/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = ResNet50(weights=weights_path, include_top=False, input_shape=(224, 224, 3))

model.trainable = False
# Add GlobalMaxPooling2D to the model to reduce the dimensionality of the output features
model = Sequential([model, GlobalMaxPooling2D()])

# Function to extract features from an image using the ResNet50 model
def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)# Convert the image to an array (necessary format for model input)
    expand_img = np.expand_dims(img_array,axis=0)# Add an additional dimension to the array (for batch size of 1)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    result = result_to_resnet.flatten() # Flatten the output feature map into a single vector
    # normalizing
    normlized_result = result / norm(result)
    return normlized_result

#print(os.listdir('fashion_small/images'))
img_files = []

for fashion_images in os.listdir('/Users/mengfy/Desktop/coding/pic/clothingdata/images'):
    images_path = os.path.join('/Users/mengfy/Desktop/coding/pic/clothingdata/images', fashion_images)
    img_files.append(images_path)

# extracting image features
image_features = []

for files in tqdm(img_files):
    features_list = extract_features(files, model)
    image_features.append(features_list)

# Save the extracted features to a file using pickle 
pickle.dump(image_features, open("image_features_embedding.pkl", "wb"))
pickle.dump(img_files, open("img_files.pkl", "wb"))
