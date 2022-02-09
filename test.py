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

img = image.load_img('sample/half.jpeg', target_size=(224, 224)) #loading image
img_array = image.img_to_array(img) #image to array
expanded_img_array = np.expand_dims(img_array, axis=0) #to batch
preprocessed_img = preprocess_input(expanded_img_array)
output = model.predict(preprocessed_img).flatten()
output_normalized = output / norm(output)
#finding distances b/w feature_list and output_normalized

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([output_normalized])
