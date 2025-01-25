import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np

# Set the background color for the app
st.markdown(
    """
    <style>
    .stApp{
        background-color: #FFCCCC;  /* Light red background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model = load_model('my_model.keras')

# Streamlit app code to upload image and classify
st.title("Blood Group Classification")

# User profile section
#st.sidebar.header("User Profile")
#profile_image = st.sidebar.file_uploader("Upload your profile image", type=["jpg", "jpeg", "png"])
#name = st.sidebar.text_input("Enter your name")

#if profile_image is not None:
 #   profile_img = Image.open(profile_image)
  #  st.sidebar.image(profile_img, caption="Profile Image", use_column_width=True)

#if name:
 #   st.sidebar.write(f"**Name:** {name}")

# Image upload and blood group classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the uploaded image for prediction
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    predicted_class = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+'][np.argmax(predictions)]
    
    st.write(f"**Predicted Blood Group:** {predicted_class}")
