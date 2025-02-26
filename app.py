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

def pdf_to_single_image(pdf_bytes):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
        
        pdf_document.close()

        # Combine all images into a single long image
        if images:
            widths, heights = zip(*(img.size for img in images))
            total_height = sum(heights)
            max_width = max(widths)

            combined_image = Image.new("RGB", (max_width, total_height), "white")
            y_offset = 0

            for img in images:
                combined_image.paste(img, (0, y_offset))
                y_offset += img.height

            return combined_image
        else:
            return None
    except Exception as e:
        st.error(f"Error converting PDF to image: {e}")
        return None
    

def analyze_image_with_gpt4v(image):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract as many user stories as possible from the given image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"}} 
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None
    

def main():
    st.title("Takim User Story Creator")
    st.write("Upload a PDF manual or document in order to receive user stories for that document using GPT-4V.") 
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        combined_image = pdf_to_single_image(uploaded_file.read())
        if combined_image:
            st.image(combined_image, caption="Combined PDF", use_column_width=True)
            analysis_result = analyze_image_with_gpt4v(combined_image)
            if analysis_result:
                st.subheader("User Stories")
                st.write(analysis_result)
            else:
                st.write("Failed to analyze document.")
        else:
            st.write("Error processing the PDF.")

if __name__ == "__main__":
    main()
