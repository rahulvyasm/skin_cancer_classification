import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('models/skin_cancer_classifier_ResNet_model.keras')

# Define the class labels corresponding to the trained model output
class_labels = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis-Like Lesions',
                'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

# Function to preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize the image to the input size expected by the model
    img = np.array(img)  # Convert the image to a NumPy array
    if img.shape[-1] != 3:  # Ensure the image has 3 color channels (RGB)
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB if needed
    img = np.expand_dims(img, axis=0)  # Add a batch dimension (1, 224, 224, 3)
    img = img / 255.0  # Normalize image (same as rescale=1./255 in your test generator)
    return img

# Function to predict the class of a given image
def predict_image(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Return predicted label and associated probabilities
    return predicted_label, prediction[0]

# Example usage (use the path to an image you want to test)
if __name__ == "__main__":
    img_path = "data/HAM10000_images_part_2/ISIC_0029313.jpg"  # Replace with the actual image path
    predicted_label, probabilities = predict_image(img_path)

    print(f"Predicted Label: {predicted_label}")
    print("Probabilities for each class:")
    for i, label in enumerate(class_labels):
        print(f"{label}: {probabilities[i]:.4f}")
