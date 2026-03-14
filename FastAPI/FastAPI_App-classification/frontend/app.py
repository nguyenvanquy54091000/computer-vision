import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

import os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/predict")


st.set_page_config(page_title="ViT Object Detection", layout="centered", page_icon="👁️")

st.title("👁️ ViT Object Detection System")
st.write("Upload an image to get predictions from the ViT model.")
st.write("Your uploaded images will be saved for continuous active learning data collection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction complete!")
                    
                    draw = ImageDraw.Draw(image)
                    bbox = result['bounding_box']
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    class_name = result['class_name']
                    confidence = result['confidence']
                    
                    font_size = max(12, int(image.height / 30))
                    try:
                        font = ImageFont.truetype("arial.ttf", size=font_size)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
                    
                    text_lines = [
                        f"Class Predicted: {class_name}",
                        f"Confidence Score: {confidence:.2%}",
                        f"BBox Details: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                    ]
                    text = "\n".join(text_lines)
                    
                    text_bbox = draw.textbbox((x1, y1), text, font=font)
                    draw.rectangle(text_bbox, fill="red")
                    draw.text((x1, y1), text, fill="white", font=font)
                    
                    st.image(image, caption="Predicted Image with Bounding Box", use_container_width=True)
                    
                    st.subheader("Results")
                    st.write(f"**Class Predicted:** `{class_name}`")
                    st.write(f"**Confidence Score:** `{confidence:.2%}`")
                    st.write(f"**Bounding Box Details:**")
                    st.json(result['bounding_box'])
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the FastAPI server. Is `python server.py` running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
