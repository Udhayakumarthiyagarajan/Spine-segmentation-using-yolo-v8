import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Set Streamlit app title
st.title("Spin Segmentation")

# Load YOLO model
model = YOLO(r"C:/Users/iproat26/Desktop/spin_segnmentation/datasets/runs/detect/train2/weights/best.pt")  # Replace with your YOLO model file if custom

# Define class names
class_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']

# File uploader for image selection
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the selected image on the left
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process and predict bounding boxes after clicking the button
    if st.button("Predict"):
        # Perform YOLO prediction
        results = model.predict(image, verbose=False)
        predictions = results[0]

        # Convert image to array for drawing
        bbox_image = np.array(image)

        # Draw bounding boxes on the image
        for box in predictions.boxes:
            # Extract coordinates, confidence, and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{class_names[class_id]}: {conf:.2f}"

            # Draw bounding box and label
            bbox_image = cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bbox_image = cv2.putText(
                bbox_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # Display the image with bounding boxes on the right
        with col2:
            st.subheader("Predicted Image")
            st.image(bbox_image, caption="Image with Bounding Boxes", use_container_width=True)
