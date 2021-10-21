import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import imutils

def VGGupdated(input_tensor= None, classes=2):
    img_rows,img_cols= 224,224
    img_channels=3
    img_dim = (img_rows,img_cols, img_channels)
    img_input = Input(shape= img_dim)

    #Block1
    x=Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu",name='block1_1')(img_input)
    x=Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu",name='block1_2')(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2),name='block1_pool')(x)

    #Block2
    x=Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name='block2_1')(x)
    x=Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name='block2_2')(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2),name='block2_pool')(x)

    #Block3
    x=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='block3_1')(x)
    x=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='block3_2')(x)
    x=Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='block3_3')(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2),name='block3_pool')(x)

    #Block4
    x=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='block4_1')(x)
    x=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='block4_2')(x)
    x=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='block4_3')(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2),name='block4_pool')(x)

    #Block5
    x=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='block5_1')(x)
    x=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='block5_2')(x)
    x=Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='block5_3')(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2),name='block5_pool')(x)

    #Classification Block
    x=Flatten()(x)
    x=Dense(4096,activation="relu",name='block6_1')(x)
    x=Dense(4096,activation="relu",name='block6_2')(x)
    x=Dense(classes, activation="softmax",name='classifications')(x)
    #classes are 2

    #Create Model
    model=Model(inputs=img_input, outputs=x, name='VGG')

    return model


model = VGGupdated(classes=2)  #2 Tumor patterns

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#appending list of images for dataframe
train_path = ''

#print(training_set.classes)
#print(training_set.class_indices)

#making directories
dataset_path = os.listdir(train_path)
Tumor_types = os.listdir(train_path)

print(Tumor_types)
print("Types of Classification: ", len(dataset_path))

Tumor = []
for item in Tumor_types:
    all_Tumor = os.listdir("Brain Tumour" + '/' +item)

#add to the list
    for tumor in all_Tumor:
       Tumor.append((item, str('Brain Tumour' + '/' +item) + '/' + tumor))
       print(Tumor)

#Building a Dataframe
Tumor_df = pd.DataFrame(data = Tumor, columns = ['Tumor type', 'image'])
print(Tumor_df.head())

#checking no. of samples for each category are present
print('Total no of classifiction in the dataset:', len(Tumor_df))
Tumor_count = Tumor_df['Tumor type'].value_counts()
print(" Ideentification in each category:")
print(Tumor_count)

# Resizing images
path = "/"
im_size = 224
images = []
labels = []
for i in Tumor_types:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold_img = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        threshold_img = cv2.erode(threshold_img, None, iterations=2)
        threshold_img = cv2.dilate(threshold_img, None, iterations=2)
        contour = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        c = max(contour, key=cv2.contourArea)

        extreme_points_left = tuple(c[c[:, :, 0].argmin()][0])
        extreme_points_right = tuple(c[c[:, :, 0].argmax()][0])
        extreme_points_top = tuple(c[c[:, :, 1].argmin()][0])
        extreme_points_bot = tuple(c[c[:, :, 1].argmax()][0])

        img = img[extreme_points_top[1]:extreme_points_bot[1], extreme_points_left[0]:extreme_points_right[0]]
        img = cv2.resize(img, dsize=(im_size, im_size))
        images.append(img)
        labels.append(i)

#Reshaping images as arrays to load the CNN model
images = np.array(images)
images = images.astype('float32')/255.0
print(images.shape)

#Arranging labels to classify in the CNN
y = Tumor_df['Tumor type'].values
y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)
print(y)

y = y.reshape(-1,1)
onehotencoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
Y = onehotencoder.fit_transform(y)
print(Y)
print(Y.shape)

#inspecting Test and Train data
train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.3, random_state=415)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

#Fitting Model
model.fit(train_x,train_y,epochs=10, batch_size=32)