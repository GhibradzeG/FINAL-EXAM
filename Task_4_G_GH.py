import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# Load gun images
gun_images_dir = "gun_images"  # Directory containing gun images
gun_images = []

for filename in os.listdir(gun_images_dir):
    img_path = os.path.join(gun_images_dir, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (32, 32))  # Resize to match CIFAR-10 image size
    gun_images.append(image)

gun_images = np.array(gun_images)

# Prepare data
x_train = x_train.astype('float32') / 255
gun_images = gun_images.astype('float32') / 255

# Create labels for guns (class 10)
y_train = np.concatenate([y_train, np.full((gun_images.shape[0],), 10)])

# Combine CIFAR-10 cars and gun images
x_train = np.concatenate([x_train, gun_images])
y_train = np.concatenate([y_train, np.full((gun_images.shape[0],), 11)])  # Label 11 for guns

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')  # 12 classes (10 for CIFAR-10 cars, 1 for guns)
])

# Compile model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
