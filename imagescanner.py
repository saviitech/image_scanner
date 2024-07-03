import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the object in the image
def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions

# Function to display the image and prediction result
def display_image_and_result(image_path, prediction):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"{prediction[1]}")
    plt.axis('off')
    plt.show()


# Function to list images in the folder
def list_images(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Main function to upload and scan image
def main():
    folder_path = 'images'
    images = list_images(folder_path)
    
    if not images:
        print("No images found in the folder.")
        return

    print("Select an image to scan:")
    for i, image_name in enumerate(images):
        print(f"{i + 1}. {image_name}")

    choice = int(input("Enter the number of the image: ")) - 1
    
    if choice < 0 or choice >= len(images):
        print("Invalid choice.")
        return

    image_path = os.path.join(folder_path, images[choice])
    predictions = predict_image(image_path)
    prediction = predictions[0]  # Take the top prediction
    print(f"{prediction[1]}: {prediction[2]*100:.2f}%")
    display_image_and_result(image_path, prediction)

if __name__ == "__main__":
    main()
