import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from gtts import gTTS

# Load a fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Function to generate caption from an uploaded image
def generate_caption(image):
    try:
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        return str(e)

# Streamlit app
def main():
    st.title("Image Captioning with Text-to-Speech")

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

            # Option to listen to the caption audio
            if st.button("Listen to Caption Audio"):
                try:
                    tts = gTTS(caption, lang='en')
                    audio_file = BytesIO()
                    tts.write_to_fp(audio_file)
                    audio_file.seek(0)
                    st.audio(audio_file, format="audio/mp3")
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
        else:
            st.warning("Please upload an image.")

    # "Refresh" button
    if st.button("Refresh"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()

