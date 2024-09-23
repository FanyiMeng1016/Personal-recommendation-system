import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt

features_list = pickle.load(open("image_features_embedding_101.pkl", "rb"))
img_files_list = pickle.load(open("img_files_101.pkl", "rb"))

print(np.array(features_list).shape)
weights_path = '/Users/mengfy/Desktop/coding/resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = ResNet101V2(weights=weights_path, include_top=False, input_shape=(224, 224,3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])

img = image.load_img('/Users/mengfy/Desktop/coding/sample/bag.png',target_size=(224,224))
img_array = image.img_to_array(img)
expand_img = np.expand_dims(img_array,axis=0)
preprocessed_img = preprocess_input(expand_img)
result_to_resnet = model.predict(preprocessed_img)
result = result_to_resnet.flatten()
# normalizing
normlized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors = 6, algorithm='brute', metric='euclidean')
neighbors.fit(features_list)
distence, indices = neighbors.kneighbors([normlized_result])
print(indices)

fig, axes = plt.subplots(1, 5, figsize=(15, 3)) # top 5 images

for i, file in enumerate(indices[0][1:6]):
    tmp_img = cv2.imread(img_files_list[file])
    tmp_img = cv2.resize(tmp_img,(200,200))
    axes[i].imshow(tmp_img)
    axes[i].axis('off')  # Turn off axis labels

plt.show()
    