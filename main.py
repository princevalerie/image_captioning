import streamlit as st
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

# Load a fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Function to generate caption from an uploaded image
def generate_caption(image):
    try:
        # Process the image and create pixel values
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        
        # Generate caption ids
        generated_ids = model.generate(pixel_values, attention_mask=None)  # You can specify attention_mask if needed

        # Decode the generated caption
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        return str(e)

# Streamlit app
def main():
    st.title("Image Captioning")

    # Upload file input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # "Generate Caption" button
    if st.button("Generate Caption"):
        if uploaded_file is not None:
            # Open the uploaded image
            image = Image.open(uploaded_file)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate caption
            caption = generate_caption(image)
            st.write("Caption:")
            st.write(caption)
        else:
            st.warning("Please upload an image.")

    # "Refresh" button
    if st.button("Refresh"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
