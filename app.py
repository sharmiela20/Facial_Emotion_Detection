import streamlit as st
from keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from PIL import Image

# Load model
model = load_model("face_exp.h5")

# Class labels
class_labels = {
    0: 'Angry',
    1: 'Happy',
    2: 'Neutral',
    3: 'Sad',
    4: 'Surprise'
}

# Function to detect faces and predict emotions
def detect_emotion(image):
    face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(image, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        preds = model.predict(face_roi)[0]
        label = class_labels[preds.argmax()]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Streamlit UI
def main():
    # Use raw string for the background image path
    background_image = "https://cdn.pixabay.com/photo/2015/05/26/23/52/technology-785742_1280.jpg"

    # Custom CSS to set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://cdn.pixabay.com/photo/2015/05/26/23/52/technology-785742_1280.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add title and uploader
    st.title("Facial Emotion Detection")
    st.write("Upload an image to instantly detect emotions with precision! Analyze facial expressions quickly and effortlessly to uncover hidden feelings in just a click.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        output_image = detect_emotion(image)
        output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        st.image(output_pil, caption="Detected Emotions", use_column_width=True)

if __name__ == "__main__":
    main()
