import streamlit as st
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor
from PIL import Image
import torch

# Load the pre-trained model, tokenizer, and image processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set the model to evaluation mode
model.eval()

# Function to generate caption
def generate_caption(image):
    # Process the image
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    # Generate the caption
    with torch.no_grad():
        output = model.generate(pixel_values, max_length=16)
    # Decode the output caption
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

# Streamlit app
st.title("Image Captioning App")
st.write("Upload an image to generate a caption.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Generate and display the caption
    if st.button("Generate Caption"):
        caption = generate_caption(image)
        st.write("Generated Caption:")
        st.write(caption)
