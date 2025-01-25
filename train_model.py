# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 classes for A, B, AB, O, A-, B-, AB-
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images to [0, 1]
    rotation_range=40,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Randomly shear images
    zoom_range=0.2  # Randomly zoom into images
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Define the train and validation generators
train_generator = train_datagen.flow_from_directory(
    r'E:\varsha_engineering\3rd_year\modifimg\train',  # Replace with your training data directory
    target_size=(128, 128),  # Resize images to match the model input size
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    r'E:\varsha_engineering\3rd_year\modifimg\validation',  # Replace with your validation data directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Save the trained model
model.save('my_model.keras')
