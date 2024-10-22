import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load the YOLOv5 model from the local repository
model = torch.hub.load('./', 'yolov5s', source='local')  # Correctly point to the local YOLOv5 directory

# Define the classes to detect
classes = ['person', 'bicycle', 'car', 'dog', 'cat']

def detect_objects(image):
    results = model(image)
    detections = results.pandas().xyxy[0]  # Get detections as a DataFrame
    return detections[detections['name'].isin(classes)]

def draw_boxes(image, detections):
    img = Image.fromarray(image)
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
        img = img.crop((xmin, ymin, xmax, ymax))
    return img

# Streamlit UI
st.title("YOLO Object Detection App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Objects"):
        image_np = np.array(image)
        detections = detect_objects(image_np)
        st.write("Detections:")
        st.dataframe(detections)

        if not detections.empty:
            # Draw bounding boxes
            image_with_boxes = draw_boxes(image_np, detections)
            st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)
        else:
            st.write("No objects detected.")
