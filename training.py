
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[24]:


import csv, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from PIL import Image

lines = []
with open('./examples/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            steering_off = 0
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                # randomly three steerting off choices
                choice = random.randint(0,2)
                source_path = batch_sample[choice]
                filename = source_path.split("/")[-1]
                current_path = './examples/IMG/' + filename
                #image = cv2.imread(current_path)
                image = Image.open(current_path).convert('RGB')
                image = np.asarray(image)
                
                ## generate steering off value
                if choice == 0:
                    steering_off = 0
                elif choice == 1:
                    steering_off = 0.1
                else:
                    steering_off = -0.1
                
                measurement = float(batch_sample[3]) + steering_off
                
                # randomly choose if use augmentation
                aug_choice = random.randint(0,1)
                
                if aug_choice == 0:
                    images.append(image)    
                    measurements.append(measurement)
                else:
                    images.append(np.fliplr(image))
                    measurements.append(-measurement)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


# In[21]:


def generator_o(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            steering_off = 0
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                #for i in range(3):
                source_path = batch_sample[0]
                filename = source_path.split("/")[-1]
                current_path = './examples/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                #if i == 0:
                #    steering_off = 0
                #elif i == 1:
                #    steering_off = 0.2
                #else:
                #    steering_off = -0.2
                measurement = float(batch_sample[3]) + steering_off
                measurements.append(measurement)
                
                #images.append(cv2.flip(image,1))
                #measurements.append(measurement*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


# In[25]:


batch_size = 64
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


# In[27]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D
import math

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

epoch = 15
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,             steps_per_epoch=math.ceil(len(train_samples)/batch_size),             validation_data=validation_generator,             validation_steps=math.ceil(len(validation_samples)/batch_size),             epochs=epoch, verbose=1)
model.save('model.h5')


# In[50]:

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')
