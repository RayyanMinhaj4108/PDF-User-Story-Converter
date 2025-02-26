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




def generate_api_code(user_story, boiler_plate="", programming_language="Python", framework="FastAPI", additional_instructions=""):
    """Generates API code based on extracted user stories using GPT-4."""
    try:
        prompt_api_creation = f"""
        ## Prompt for API Code Generation (Code-Only Output)

        **Instructions:**

        You are an AI code generation assistant. Your task is to generate the API code given the boilerplate, and given detailed user story and acceptance criteria, programming language, API framework, and additional instructions. Do *not* include any explanatory text, comments outside of the code itself, or any other information besides the code. Ensure the generated code is well-structured, readable, and follows best practices for the chosen language and framework. Include necessary error handling and consider security implications where applicable. Assume all necessary libraries and dependencies are pre-installed. Focus on providing a functional API implementation.

        Also if any API is not implemented in the boilerplate, please implement it in the final code given the User story and Gherkin.

        **Input:**

        1. **Boilerplate Code:**
        {boiler_plate}

        2. **User Story and Gherkin:** 
        {user_story}

        3. **Programming Language:**
        {programming_language}

        4. **API Framework:** 
        {framework}

        If the provided framework has issues or is not provided, then choose the best framework for the given programming language.

        5. **Additional Instructions:**
        {additional_instructions}

        **Output:**
        Provide only the complete and functional API code in the specified programming language and framework. Include necessary imports, function definitions, routing, middleware (if applicable), and any other required code.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_api_creation}],
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating API code: {e}")
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

            # Generate API code based on extracted user stories
            st.subheader("Generated API Code")
            boiler_plate = st.text_area("Boilerplate Code (Optional)", "")
            programming_language = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C#", "Go"])
            framework = st.text_input("Preferred API Framework", "FastAPI")
            additional_instructions = st.text_area("Additional Instructions (Optional)", "")

            if st.button("Generate API Code"):
                api_code = generate_api_code(analysis_result, boiler_plate, programming_language, framework, additional_instructions)
                if api_code:
                    st.code(api_code, language=programming_language.lower())
                else:
                    st.write("Failed to generate API code.")
        else:
            st.write("Failed to analyze the image.")

if __name__ == "__main__":
    main()
