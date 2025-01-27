import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('models/skin_cancer_classifier_ResNet_model.h5',
                   custom_objects={'CustomScaleLayer': CustomScaleLayer})



def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values
    return img_array

# Function to make predictions
def make_prediction(img_path):
    # Preprocess the image
    processed_image = preprocess_image(img_path)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)

    # Map predicted class to the actual labels
    class_labels = ['Melanoma', 'Benign Keratosis-Like Lesions', 'Basal Cell Carcinoma',
                    'Actinic Keratoses', 'Vascular Lesions', 'Dermatofibroma', 'Melanocytic Nevi']
    prediction_label = class_labels[predicted_class[0]]

    # Display the image and the prediction
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {prediction_label}")
    plt.show()

    return prediction_label

# Example Usage
image_path = 'data/HAM10000_images_part_2/ISIC_0029308.jpg'  # Replace with the path to your image
prediction = make_prediction(image_path)
print(f"Predicted class: {prediction}")
