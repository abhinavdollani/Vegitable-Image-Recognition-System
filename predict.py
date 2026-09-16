import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model(
    "model/vegetable_model.keras"
)

# Class names
class_names = [
    "cabbage",
    "carrot",
    "onion",
    "potato",
    "tomato"
]

# Image to predict
image_path = "test.jpg"

# Load image
img = load_img(
    image_path,
    target_size=(180, 180)
)

# Convert image to array
img_array = img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
predictions = model.predict(img_array)

# Get highest probability
predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

print("--------------------------------")
print("Vegetable Recognition Result")
print("--------------------------------")
print("Vegetable:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))