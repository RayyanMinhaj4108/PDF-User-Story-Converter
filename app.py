import streamlit as st
import openai
import io
import base64
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def analyze_image_with_gpt4v(image):
    prompt= """
Identify any fields, buttons and text in the screenshots and create user stories with acceptance criteria in BDD/Gherkin format from them

Display Results as follows:

#### **User Story:**
#     As a [role], I want [feature] so that [benefit].

#     #### **Acceptance Criteria:**
#     - **Feature:** A brief description of the functionality
#       - **Scenario:** Provide a detailed name for each scenario.
#       - **Given:** Outline the preconditions necessary for the scenario.
#       - **When:** Specify the actions taken by the user.
#       - **Then:** State the expected results after the actions.    

Requirements:
-DO NOT WRITE ETC, WRITE IN DETAIL AND WRITE FULL SENTENCES
-Also ensure all text fields and buttons and text are mentioned in the user stories and acceptance criteria.
-Do not use short forms or reduce the number of examples; include every option explicitly.
-Write out all details completely without omitting any examples or categories.
"""

    try:
        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        # Send image to OpenAI's GPT-4V
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"}},
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
    st.write("Upload an image to receive user stories using GPT-4V.") 
    
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Open image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze image
        analysis_result = analyze_image_with_gpt4v(image)
        if analysis_result:
            st.subheader("Extracted User Stories")
            st.write(analysis_result)
        else:
            st.write("Failed to analyze the image.")

if __name__ == "__main__":
    main()
