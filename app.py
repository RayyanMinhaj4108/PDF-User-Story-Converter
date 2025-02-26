import streamlit as st
import openai
import fitz
import io
import base64
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def pdf_to_images(pdf_bytes):
    images = []
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return None
    

def analyze_images_with_gpt4v(images):
    try:
        image_contents = []
        
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"}})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract as many user stories as possible from the given images."}
                    ] + image_contents,  # Pass all images at once
                }
            ],
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing images: {e}")
        return None
    

def main():
    st.title("Takim User Story Creator")
    st.write("Upload a PDF manual or document in order to receive user stories for that document using GPT-4V.") 
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        images = pdf_to_images(uploaded_file.read())
        if images:
            for i, image in enumerate(images):
                st.image(image, caption=f"Page {i + 1}", use_column_width=True)
            
            analysis_result = analyze_images_with_gpt4v(images)
            if analysis_result:
                st.subheader("User Stories")
                st.write(analysis_result)
            else:
                st.write("Failed to analyze document.")
        else:
            st.write("Error processing the PDF.")

if __name__ == "__main__":
    main()
