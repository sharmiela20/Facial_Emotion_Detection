import os
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import gdown

# Function to download the model from Google Drive
def download_model_from_gdrive():
    # Google Drive file ID and URL
    file_id = '1IXAEpGKGdSLCGS4uCl3YA-FaOwtfKgoL'
    gdrive_url = f'https://drive.google.com/uc?id={file_id}'

    # Output model path
    model_output_path = r'./models/face_exp.h5'

    # Ensure the './models' directory exists
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Download the model if not already present
    if not os.path.exists(model_output_path):
        gdown.download(gdrive_url, model_output_path, quiet=False)

    return model_output_path

# Download the model from Google Drive
model_path = download_model_from_gdrive()

# Load the pre-trained model
model = load_model(model_path)

# Class labels for emotion prediction
class_labels = {
    0: 'Angry',
    1: 'Happy',
    2: 'Neutral',
    3: 'Sad',
    4: 'Surprise'
}

# Function to detect faces and predict emotions
def detect_emotion(image):
    # Load Haar cascade classifier for face detection
    face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(image, 1.3, 5)

    # Process each face detected
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = image[y:y+h, x:x+w]

        # Resize face ROI to the required input size of the model
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Make prediction using the model
        preds = model.predict(face_roi)[0]
        label = class_labels[preds.argmax()]

        # Draw bounding box and label on the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Streamlit UI
def main():
    # Custom CSS for Streamlit UI to enhance appearance
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .upload-btn-wrapper {
            display: flex;
            justify-content: center;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and Description
    st.title("Facial Expression Detection")
    st.markdown("Upload an image to detect **facial expressions**. Our model can detect a variety of expressions with high accuracy!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display instructions
    st.markdown("Make sure the uploaded image contains a clear face for the best results.")

    if uploaded_file is not None:
        # Read the image from uploaded file
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Detect emotions
        output_image = detect_emotion(image)

        # Convert the output image to RGB (for displaying in Streamlit)
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        # Display the output image
        st.image(output_image_rgb, caption="Detected Emotions", use_column_width=True)

        # Optionally provide a download button for the image
        st.download_button(
            label="Download the result image",
            data=cv2.imencode('.jpg', output_image)[1].tobytes(),
            file_name="emotion_detected_image.jpg",
            mime="image/jpeg"
        )

# Run the Streamlit app
if __name__ == "__main__":
    main()
