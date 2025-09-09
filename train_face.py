import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 👇 Yahan apna dataset ka path set karo
dataset_path = r"E:\MultimodalPrototype\face_dataset"

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    dataset_path + "\\train",
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    dataset_path + "\\test",
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical"
)

# Simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(48,48,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=test_generator, epochs=10)

# Save the model
model.save("face_emotion_model.h5")
print("✅ Face model training complete and saved as face_emotion_model.h5")
