import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv(r"A_Z Handwritten Data.csv").astype('float32') # reading dataset 
print(data.head(10)) # printing the first 10 images for now

# Split the data into images and their labels. 
X = data.drop('0', axis=1) 
y = data['0']

# Reshape the data from the csv file so it can display as an image content. 
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)

train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28)) # converting to 28x28 pixels
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28)) # converting to 28x28 pixels
print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

# Then plotting the number of alphabets in the datasets 
y_int = np.int0(y)
count = np.zeros(26, dtype='int')
for i in y_int:
    count[i] +=1
alphabets = [] # creats a list for the alphabets containing all the characters using the value() function. 
for i in word_dict.values():
    alphabets.append(i)
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.barh(alphabets, count) # using the count and alphabets to draw a horizontal bar plot
plt.xlabel("Count the Number of elements ") # label for x axis
plt.ylabel("The Alphabets") # label for y axis
plt.grid()
plt.show()

# Then we shuffle the data 
shuff = shuffle(train_x[:100])  # shuffles some of the image from the train area. 

fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()
for i in range(9): # then creates a 9 plot in 3x3 shaped to display the thresholded images of 9 alphabets (as of now)
    _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()

# Data reshaping -- meaning reshaping the training and test datasets so we can create our model. 
train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("The new shape of train data: ", train_X.shape)
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("The new shape of train data: ", test_X.shape)

# next we need to convert from float values to categorical values. 
train_yOHE = to_categorical(train_y, num_classes = 26, dtype='int')
print("The new shape of train labels: ", train_yOHE.shape)
test_yOHE = to_categorical(test_y, num_classes = 26, dtype='int')
print("The new shape of test labels: ", test_yOHE.shape)

# Making CNN to work in this process
model = Sequential() # from keras models

# Design the CNN model over our training dataset 
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(26,activation ="softmax"))

# Compile our model
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit our model
history = model.fit(train_X, train_yOHE, epochs=1,  validation_data = (test_X,test_yOHE))

model.summary()
model.save('handW_model.h5')

# We want to print our train and validation accuracies and or even our losses
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])

# Lets do some prediction on test data 
fig, axes = plt.subplots(3,3, figsize=(8,9))
axes = axes.flatten()

for i, ax in enumerate(axes): 
  img = np.reshape(test_X[i], (28,28))
  ax.imshow(img, cmap="Greys")

  pred = word_dict[np.argmax(test_yOHE[i])]
  ax.set_title("Prediction: "+pred) 
  ax.grid()

 
# Converts Keras (.h5) model to CoreML (.mlmodel)
import coremltools
# TODO: THE ERROR IS BELOW CODE....!!!
coreml_model = coremltools.converters.keras.convert('/Users/vartanarzumanyan/CSMA.213-Artificial Intelligence/Final/handW_model.h5')
from keras.models import load_model
model = load_model('/Users/vartanarzumanyan/CSMA.213-Artificial Intelligence/Final/handW_model.h5')
coreml_model.save('handwritingRecogniton.mlmodel')
coreml_model.save_weight()

 

 