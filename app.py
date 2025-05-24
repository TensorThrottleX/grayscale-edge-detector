# -- coding: utf-8 --

import streamlit as st
import requests
import numpy as np
from io import BytesIO
from PIL import Image  
from skimage import filters


# Check if the input is a URL or local file
def is_url(path):
    return path.startswith('http://') or path.startswith('https://')

# Load image from URL or file path
def load_image(path_or_url):
    if is_url(path_or_url):
        response = requests.get(path_or_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(path_or_url)

# Streamlit App
def main():
    st.title("Grayscale & Edge Detection App")

    # User input
    path_or_url = st.text_input("Throw your image URL or upload file below:")
    uploaded_file = st.file_uploader("...or upload a local image file", type=['jpg', 'png', 'jpeg'])

    quality = st.slider("JPEG save quality", 1, 100, 85)

    if st.button("Bang Your Image"):
        try:
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
            elif path_or_url:
                img = load_image(path_or_url)
            else:
                st.warning("Please provide a valid image URL or upload a file.")
                return

            # Grayscale
            grey_img = img.convert('L')

            # Edge Detection
            gray_array = np.array(grey_img)
            edges = filters.sobel(gray_array)
            edges_8bit = (edges * 255).astype(np.uint8)
            edges_img = Image.fromarray(edges_8bit)

            # Display images
            st.subheader("Grayscale Image")
            st.image(grey_img, use_column_width=True)

            st.subheader("Edge Detection")
            st.image(edges_img, use_column_width=True)

            # Download buttons
            # Grayscale
            grey_buffer = BytesIO()
            grey_img.save(grey_buffer, format="JPEG", quality=quality)
            grey_buffer.seek(0)
            st.download_button(
                label="Download Grayscale Image",
                data=grey_buffer,
                file_name="grayscale.jpg",
                mime="image/jpeg"
            )

            # Edge-detected
            edges_buffer = BytesIO()
            edges_img.save(edges_buffer, format="JPEG", quality=quality)
            edges_buffer.seek(0)
            st.download_button(
                label="Download Edge-Detected Image",
                data=edges_buffer,
                file_name="edges.jpg",
                mime="image/jpeg"
            )

            # Save images locally (optional for dev use)
            grey_img.save("grey_image.jpg", quality=quality)
            edges_img.save("edges_image.jpg", quality=quality)

            st.success(f"Images saved with quality = {quality}.")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
