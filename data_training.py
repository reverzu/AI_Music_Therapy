import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 
from keras.models import Model
 
# Initialization
is_init = False
# Tensor Size
size = -1

# Labels
label = []
# Labels Dictionary
dictionary = {}
c = 0

# Reading numpy labels from current directory
for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1

# Mapping facial data with emotions from the Numpy Array
for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

# Dividing data to categories
y = to_categorical(y)

# Making new copies of X and Y coordinates
X_new = X.copy()
y_new = y.copy()
counter = 0 


# Arranging shape of Array content
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1

# Shaping Data Input
ip = Input(shape=(X.shape[1]))

# Starting the neural network - Input Layer
m = Dense(512, activation="relu")(ip)
# Neural network - Middle Layer
m = Dense(256, activation="relu")(m)

# Neural network - Output Layer
op = Dense(y.shape[1], activation="softmax")(m) 

# Model Creation
model = Model(inputs=ip, outputs=op)

# Model Compilation
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Model Fitting
model.fit(X, y, epochs=50)

# Model Saving
model.save("model.h5")
# Saving model Emotions
np.save("labels.npy", np.array(label))
