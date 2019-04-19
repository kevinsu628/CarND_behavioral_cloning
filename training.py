
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt


# In[5]:


import csv, cv2
import numpy as np

lines = []
with open('./examples/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split("/")[-1]
        current_path = './examples/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


# In[6]:


augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# In[15]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D

model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Conv2D(24, (2, 2), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (2, 2), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (2, 2), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (1, 1), activation="relu"))
model.add(Conv2D(64, (1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model.h5')

