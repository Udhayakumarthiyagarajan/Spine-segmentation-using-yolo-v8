import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Set Streamlit app title
st.title("Spine Segmentation Using Yolov8")

# Load YOLO model
model = YOLO(r"datasets/runs/detect/train2/weights/best.pt")  # Replace with your YOLO model file if custom

# Define class names and assign a unique color for each class
class_names = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']
class_colors = {
    'L1': (255, 0, 0),   # Red
    'L2': (0, 255, 0),   # Green
    'L3': (0, 0, 255),   # Blue
    'L4': (255, 255, 0), # Cyan
    'L5': (255, 0, 255), # Magenta
    'S1': (0, 255, 255)  # Yellow
}

# File uploader for image selection
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the selected image on the left
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selected Image")
        image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process and predict bounding boxes after clicking the button
    if st.button("Predict"):
        # Perform YOLO prediction
        results = model.predict(image, verbose=False)
        predictions = results[0]

        # Convert image to RGB NumPy array for drawing
        bbox_image = np.array(image)  # RGB NumPy array

        # Draw bounding boxes on the image
        for box in predictions.boxes:
            # Extract coordinates, class ID, and confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            color = class_colors[class_name]
            confidence = box.conf[0]

            # Draw bounding box
            bbox_image = cv2.rectangle(bbox_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class name and confidence score in bold
            bbox_image = cv2.putText(
                bbox_image, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA
            )

        # Display the image with bounding boxes on the right
        with col2:
            st.subheader("Predicted Image")
            st.image(bbox_image, caption="Image with Bounding Boxes", use_container_width=True)
