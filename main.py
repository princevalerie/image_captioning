import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests

# Load model, processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate captions with attention mask
def generate_caption(image, max_length=16, num_beams=4):
    # Preprocess the image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate attention mask
    attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long, device=device)

    # Generate the output sequence
    output_ids = model.generate(
        pixel_values,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )

    # Decode the generated sequence
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Streamlit app
st.title("Image Captioning with Attention Mask")
st.write("Upload an image and the model will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
        st.write("Generated Caption:", caption)
