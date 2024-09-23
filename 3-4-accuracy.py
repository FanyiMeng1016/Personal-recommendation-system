import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt

features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

print(np.array(features_list).shape)

weights_path = '/Users/mengfy/Desktop/coding/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = ResNet50(weights=weights_path, include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])

img = image.load_img('/Users/mengfy/Desktop/coding/sample/shoes.png',target_size=(224,224))
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

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, file in enumerate(indices[0][1:6]):
    tmp_img = cv2.imread(img_files_list[file])
    tmp_img = cv2.resize(tmp_img,(200,200))
    axes[i].imshow(tmp_img)
    axes[i].axis('off')  # Turn off axis labels

plt.show()

correct_image = '/Users/mengfy/Desktop/coding/pic/clothingdata/images/14332.jpg'  #Ground Truth 
correct_index = img_files_list.index(correct_image)

k = 5
top_k_indices = indices[0][1:k+1]

# Define Top-k Accuracy
def top_k_accuracy(y_true, y_pred, k=6):
    correct = 0
    
    for true_label, pred_labels in zip(y_true, y_pred):
        if true_label in pred_labels[:k]:
            correct += 1
    return correct / len(y_true)

y_true = [correct_index]  
y_pred = [top_k_indices]   

# calculate Top-k Accuracy
accuracy = top_k_accuracy(y_true, y_pred, k=k)
print(f"Top-{k} Accuracy: {accuracy:.2f}")

scores = np.array([0.2, 0.5, 0.3, 0.8, 0.4])
top_k_indices = np.argsort(scores)[-k:][::-1]
top_k_scores = scores[top_k_indices]
print("Top-k scores:", top_k_scores)
print("Top-k indices:", top_k_indices)
