import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from PIL import Image
import requests

# Load the model and tokenizer
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Configure model to avoid warnings
if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

# Streamlit app
st.title("Image Captioning App")
st.write("Upload an image and get a caption generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def predict_caption(image):
    # Preprocess the image
    image = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate the caption
    generated_ids = model.generate(image, max_length=50, num_beams=4, no_repeat_ngram_size=2)
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return caption

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the caption
    caption = predict_caption(image)
    st.write(f"**Generated Caption:** {caption}")
