import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model = load_model('/content/drive/MyDrive/FinalProject/model.keras')

# Class map (update if needed)
class_map = {
    'Tomato___Bacterial_spot': 0,
    'Tomato___Early_blight': 1,
    'Tomato___Late_blight': 2,
    'Tomato___Leaf_Mold': 3,
    'Tomato___Septoria_leaf_spot': 4,
    'Tomato___Spider_mites': 5,
    'Tomato___Target_Spot': 6,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 7,
    'Tomato___Tomato_mosaic_virus': 8,
    'Tomato___Healthy': 9,
    'Unknown': 10
}
reverse_map = {v: k for k, v in class_map.items()}

# Image preprocessing
def preprocess_image(img, img_size=128):
    if len(img.shape) == 2: # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Prediction function
def predict(img):
    # Convert the image to HSV color space for better color analysis
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a range for green color in HSv
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Calculate the percentage of green pixels
    green_percentage = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    # Set a threshold for green pigment
    green_threshold = 0.10  # 10% green pixels
    if green_percentage < green_threshold:
        return "Unknown (Not a Leaf)", 0.0

    # If enough green is detected, proceed with the model prediction
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    confidence = np.max(prediction)
    class_idx = np.argmax(prediction)
    class_name = reverse_map[class_idx]

    # Set a confidence threshold for classifying as 'Unknown' based on model prediction
    confidence_threshold = 0.6

    # If the predicted class is 'Unknown' or the confidence is below the threshold
    if class_name == "Unknown" or confidence < confidence_threshold:
        # Return "Unknown" without the "Not a Leaf" suffix if it's due to low model confidence
        return "Unknown", confidence
    else:
        return class_name, confidence


# Streamlit UI
st.set_page_config(page_title="Tomato Disease Detector", layout="centered")
st.title("🍅 Tomato Disease Detection App")


# Display Test Accuracy
test_accuracy = 0.7310 # Using the actual test accuracy from the notebook output
st.markdown(f"### Test Accuracy: `{test_accuracy:.4f}`")

# Display Plots
epochs = range(1, 11)
train_acc = [0.2803, 0.5174, 0.6494, 0.7283, 0.7993, 0.8664, 0.8618, 0.9029, 0.9559, 0.9554] # Using example data from notebook
val_acc = [0.4269, 0.6023, 0.5497, 0.6082, 0.6725, 0.6199, 0.6316, 0.6842, 0.6667, 0.6608] # Using example data from notebook
train_loss = [2.0666, 1.3774, 0.9950, 0.7938, 0.5992, 0.4085, 0.4041, 0.2760, 0.1564, 0.1413] # Using example data from notebook
val_loss = [1.4720, 1.1643, 1.3743, 1.2337, 1.0788, 1.3033, 1.4098, 1.2327, 1.9072, 1.6141] # Using example data from notebook


plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
st.pyplot(plt) # Display the plot in Streamlit
plt.close() # Close the plot figure


st.markdown("---") # Separator

st.markdown("### Image Prediction")

# Add selectbox to sidebar for choosing input method
input_method = st.sidebar.selectbox("Choose input method:", ("Upload Image", "Take Photo"))

img = None # Initialize img to None

if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) # Read as color image
        st.image(img, caption="Uploaded Image", use_column_width=True, channels="BGR")

elif input_method == "Take Photo":
    camera_image = st.sidebar.camera_input("Take a photo...")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) # Read as color image
        st.image(img, caption="Captured Image", use_column_width=True, channels="BGR")

# Add a button to trigger prediction
if img is not None and st.button("Make Prediction"):
    label, confidence = predict(img)
    st.markdown(f"### 🧠 Prediction: `{label}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")
