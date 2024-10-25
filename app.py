import streamlit as st
from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import io

# Load model and processor
model_id = "Salesforce/blip-image-captioning-base"
model = BlipForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

def generate_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Streamlit app layout
st.title("Image Captioning App")
st.write("Upload an image to get a generated caption.")

# Image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Generate and display caption
    caption = generate_caption(uploaded_image.read())
    st.write("Generated Caption:")
    st.write(caption)
