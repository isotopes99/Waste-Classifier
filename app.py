import streamlit as st
import cv2 
import numpy as np 
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Load the model

@st.cache_resource
def load_my_model():
    return load_model("best_model.keras")
# load the model ( only once not on every rerun)
model = load_my_model()

# Target size initialization
target_size = (224, 224)

# Function to preprocess image data
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    image = pil_image.resize(target_size)
    image_array = np.array(image) / 255.0 
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function predicting the classification
def predict(image_array: np.ndarray) -> str:
    prediction = model.predict(image_array)[0][0]
    class_idx = np.argmax(prediction)
    classes = ["Recyclable" , "Organic"]
    return classes[class_idx]

# Streamlit app
st.title("Waste Classification APP")
mode = st.sidebar.selectbox("Select Mode", ["Image Classification", "Real-Time Video"])

if mode == "Image Classification":
    st.header("Classify an Image")

    option = st.radio("Choose input method:", ("Upload Image", "Capture Image"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image_array = preprocess_image(image)
            result = predict(image_array)
            st.write("Prediction:", result)
        else:
            st.warning("Please upload an image.")

    elif option == "Capture Image":
        cam_image = st.camera_input("Take a Picture")
        if cam_image is not None:
            image = Image.open(cam_image)
            st.image(image, caption="Captured Image", use_column_width=True)
            image_array = preprocess_image(image)
            result = predict(image_array)
            st.write("Prediction:", result)
        else:
            st.warning("Please capture an image.")

elif mode == "Real-Time Video":
    st.header("Real-Time Video Classifier")

    # Initialize session state for video classification
    if "run_video" not in st.session_state:
        st.session_state.run_video = False

    # Start/Stop buttons for controlling video
    start_video = st.checkbox("Start Video Classification", key="start_video_checkbox")

    if start_video:
        st.session_state.run_video = True
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam")
        else:
            video_placeholder = st.empty()
            

            while st.session_state.run_video:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                # Predict classification
                image_array = preprocess_image(pil_frame)
                result = predict(image_array)

                # Display classification result on the video frame
                cv2.putText(frame_rgb, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Display frame in Streamlit
                video_placeholder.image(frame_rgb, channels="RGB")

                # Stop condition

                time.sleep(0.03)
            
                


            cap.release()
            
    else:
        st.write("Tick the checkbox to start real-time video classification.")
