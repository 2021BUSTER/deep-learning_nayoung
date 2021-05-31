import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt


# Set your dataset directory
# Directory Structure:
# -- train-set
# ------------/on_mask
# ------------/off_mask
# --- test-set
# ------------/on_mask
# ------------/off_mask

TRAINING_DIR = "/Users/jeikei/Documents/deep_summer_dataset/train-set"
VALIDATION_DIR = "/Users/jeikei/Documents/deep_summer_dataset/test-set"

batch_size = 8

# Image Data Generator with Augmentation
training_datagen = ImageDataGenerator(
      rescale=1./255, # 0~1 사이의 값으로 만들어줌
      width_shift_range=0.2, # 이미지를 좌우로 움직임
      height_shift_range=0.2, # 이미지를 상하로 움직임
      brightness_range=(0.5, 1.3), # 50퍼센트 어둡게, 30퍼센트 밝게
      )

validation_datagen = ImageDataGenerator(rescale=1./255)

# Reading images from directory and pass them to the model
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,  # 내부 폴더에 있는 이미지도 각각 읽어옴
    batch_size=batch_size, # 한번에 몇개 읽을 지
    target_size=(224, 224), # input 이미지 사이즈
    class_mode='categorical', # 폴더의 리스트를 쭉 읽어서 개수만큼 클래스 생성
    shuffle=True # 데이터의 순서를 랜덤하게 읽어오는 것
)

# test set
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical'
)

# Plotting the augmented images
# 
img, label = next(train_generator)
plt.figure(figsize=(20, 20))

for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(img[i])
    plt.title(label[i])
    plt.axis('off')

plt.show()

# Load pre-trained base model.
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False, weights='imagenet')
# Freeze the base model
base_model.trainable = False

# Add Custom layers
out_layer = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', activation=None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer) # 7x7x128

out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer) # 128

out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)

# Make New Model
model = tf.keras.models.Model(base_model.input, out_layer)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Training
history = model.fit(train_generator, epochs=25,
                    validation_data=validation_generator, verbose=1)

# Save the trained model
model.save("saved_model.h5")