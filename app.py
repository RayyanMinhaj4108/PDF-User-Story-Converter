import streamlit as st
import openai
import io
import base64
from PIL import Image
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF

# import google.generativeai as genai
from google import genai

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


client = openai.OpenAI(api_key=OPENAI_API_KEY)


def extract_images_from_pdf(pdf_file):  # Removed the output_folder parameter
    """Extracts images from a PDF file and returns a list of PIL Image objects."""
    doc = fitz.open(stream=pdf_file, filetype="pdf")

    images = []  # Store PIL Image objects

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]  # Image reference
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # image_ext = base_image["ext"] # Removed the image extension

            # Create a PIL Image object from the bytes
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    print(f"Images extracted successfully.")
    return images  # Return the list of PIL Image objects


def analyze_image_with_gpt4v(image):
    """Analyzes an image with GPT-4V."""
    prompt = """
    Identify any fields, buttons and text in the screenshots and create user stories with acceptance criteria in BDD/Gherkin format from them.

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
    -Write out all details completely without omitting any examples or categories; include ALL of them.
    -Write out all details completely without omitting any examples or categories; include ALL of them.
    """

    try:
        if isinstance(image, str):
            # If image is a path, open it with PIL
            pil_image = Image.open(image)
        else:
            # If image is already a PIL Image object, use it directly
            pil_image = image

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Send image to OpenAI's GPT-4V
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None


def generate_orm_context(database, orms, programming_language):
    """Generates context for ORM usage based on database and ORM choice."""

    if database == "JSON":
        return "There are no ORMS, just store the data in a Dictionary or JSON."  # No ORMs for JSON

    if orms == "YES":
        if database == "SQLite":
            if programming_language.lower() == "python":
                return (
                    "Use the best suited ORM for the given database, such as SQLAlchemy"
                )
            elif programming_language.lower() == "javascript":
                return "Use the best suited ORM for the given database, such as Sequelize or TypeORM with a SQLite driver."
            elif programming_language.lower() == "java":
                return "Use the best suited ORM for the given database, such as Hibernate or jOOQ with a SQLite dialect."
            else:
                return "Use the best suited ORM for the given database."

        elif database == "Postgres":
            if programming_language.lower() == "python":
                return "Use the best suited ORM for the given database, such as SQLAlchemy or Django ORM."
            elif programming_language.lower() == "javascript":
                return "Use the best suited ORM for the given database, such as Sequelize, TypeORM, or Prisma."
            elif programming_language.lower() == "java":
                return "Use the best suited ORM for the given database, such as Hibernate or jOOQ."
            else:
                return "Use the best suited ORM for the given database."
        else:
            return "Use the best suited ORM for the given database."

    elif orms == "NO":
        return "Do not use ORMS and ensure raw SQL statements are used."
    else:
        return "Invalid ORM option."


def generate_boilerplate(
    user_stories,
    programming_language="Python",
    framework="Flask",
    additional_instructions="",
    database="JSON",
    orms="NO",
    yaml_generated=""
):
    """Generates boilerplate API code based on extracted user stories using GPT-4."""
    try:
        orm_context = generate_orm_context(database, orms, programming_language)

        prompt_boiler_plate_creation = f"""
        ## Prompt for API Boilerplate Code Generation

        **Instructions:**
        You are an AI code generation assistant. Your task is to generate *only* the boilerplate code of the API based on the provided user stories, programming language, API framework, Database, ORMs (if any), and the actual schema in YAML format, and additional instructions. Do *not* include any explanatory text, comments outside of the code itself, or any other information besides the code. Ensure the generated code is well-structured, readable, and follows best practices for the chosen language and framework. Include necessary error handling and consider security implications where applicable. Assume all necessary libraries and dependencies are pre-installed. Focus on providing a functional boilerplate for an API implementation.
        

        **Important Instructions**
        - Further ensure that the API includes CORS policy with allow-all origins. And further include any other security considerations.
        - Further ensure there are GET requests for each respective POST request.

        **Input:**
        1. **User Stories:**
        {user_stories}

        2. **Programming Language:**
        {programming_language}

        3. **API Framework:**
        {framework}

        If the provided framework has issues or is not provided, then choose the best framework for the given programming language.
        
        4. **Database:**
        {database}
        
        5. **ORM:**
        {orm_context}

        6. **YAML Schema:**
        {yaml_generated}
        
        7. **Additional Instructions:**
        {additional_instructions}

        **Output:**
        Provide *only* the boilerplate code with the stated database above defined and API definitions.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_boiler_plate_creation}],
            max_tokens=2000,
        )
        return response.choices[0].message.content, prompt_boiler_plate_creation

    except Exception as e:
        st.error(f"Error generating boilerplate code: {e}")
        return None


def generate_api_code(
    user_story,
    boiler_plate="",
    programming_language="Python",
    framework="Flask",
    additional_instructions="",
    combined_user_stories="",
    database="PostgreSQL",
    orms="SQLAlchemy",
    yaml_generated="",
):
    """Generates API code based on extracted user stories using Gemini 2.0 Flash, with context."""
    try:
        context = ""
        if combined_user_stories:
            # context = f"""
            # Previous Image User Stories and Acceptance Criterias:
            # {combined_user_stories}

            # Previous Generated Code that you need to build upon:
            # {previous_code}
            # """

            context = f"""
            Previous Image User Stories and Acceptance Criterias that were used to build the code provided below:
            {combined_user_stories}
            """

        prompt_api_creation = f"""
        ## Prompt for API Code Generation (Code-Only Output)

        **Instructions:**

        You are an AI code generation assistant. Your task is to generate the API code given the boilerplate, detailed user story and acceptance criteria, programming language, API framework, and additional instructions.  Consider the previous context of user stories and code when generating the API for this current user story.  Do *not* include any explanatory text, comments outside of the code itself, or any other information besides the code. Ensure the generated code is well-structured, readable, and follows best practices for the chosen language and framework. Include necessary error handling and consider security implications where applicable. Assume all necessary libraries and dependencies are pre-installed. Focus on providing a functional API implementation.
        Replace any comments that says "Simulate the logic by yourself", "Add the logic by yourself", etc. These type of comments should be replaced with the correct implementation based on the userstories and best practices.
        If the code is using raw dictionaries and manual data retrieval/storage, replace them with {orms}-based queries while following {database}'s best practices.
        
        Also if any API is not implemented in the boilerplate, please implement it in the final code given the User story and Gherkin.

        **Important Instructions:**
        - Avoid reusing similar logic, instead use functions or reusable components where necessary. 
        - Ensure the code remains within 800 lines, avoiding unnecessary verbosity while maintaining readability and completeness.
        - Further ensure that the API includes CORS policy with allow-all origins. And further include any other security considerations.
        - Further ensure there are GET requests for each respective POST request.
        - Ensure the outputs are not mock-up dictionaries but actual data from the database.

        {context}  <-- Previous context

        **Input:**

        1. **Previously produced Code that you need to build upon:**
        This code contains placeholder comments and logic that needs to be replaced with the implementation.
        {boiler_plate}

        2. **User Story and Gherkin:**
        {user_story}

        3. **Programming Language:**
        {programming_language}

        4. **API Framework:**
        {framework}

        5. **Database**
        {database}
        
        6. **ORM**
        {orms}

        7. **YAML Schema:**
        {yaml_generated}

        If the provided framework has issues or is not provided, then choose the best framework for the given programming language.

        8. **Additional Instructions:**
        {additional_instructions}

        **Expected Output:**
        1. Provide only the complete and functional API code in the specified programming language and framework. Include necessary imports, function definitions, routing, middleware (if applicable), and any other required code.
        2. A complete, functional version of the given code with placeholder comments replaced by actual logic.
        3. Any necessary ORM-based data handling instead of dictionary-based storage.
        4. Proper adherence to {framework} best practices.
        
        """

        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_api_creation,
            config={
                "temperature": 0.3,  # 0 to 2
                "top_p": 0.9,
                "top_k": 50,
            },
        )

        # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # model = genai.GenerativeModel('gemini-2.0-flash') # Using Gemini 1.5 Flash

        # response = model.generate_content(prompt_api_creation, request_options={"timeout":1000})

        return (
            response.text,
            response.usage_metadata.candidates_token_count,
            prompt_api_creation,
        )

    except Exception as e:
        st.error(f"Error generating API code: {e}")
        return None


def generate_yaml_schema(
    user_story,
    combined_user_stories,
    previous_yaml_schema="",
):
    """Generates yaml schema based on extracted user stories using Gemini 2.0 Flash, with context."""
    try:
        system_prompt = """
        You are an expert database architect, specializing in creating practical, efficient relational schemas.
        Your priority is to ensure the schema is normalized and clean, but avoid overcomplication, especially with unnecessary junction tables.
        """

        context = ""
        if combined_user_stories:
            context = f"""
            Previous Image User Stories and Acceptance Criterias that were used to generate the current schema:
            {combined_user_stories}
            """

    #     prompt_schema_creation = f"""
    #     ## Prompt for Schema Code Generation (YAML)

    #    You are an expert database schema designer. Your task is to generate a well-structured, normalized (3NF) relational database schema in YAML format based on provided user stories and their corresponding acceptance criteria, I have also generated the previously generated yaml schema for reference so please edit it and add to it. Focus on creating a schema that minimizes redundancy, ensures data integrity, and avoids excessive columns per table.

    #     **Instructions:**

    #     1.  **Analyze User Story and Acceptance Criteria:** Carefully analyze the provided user story and its acceptance criteria to identify all entities, attributes, and relationships necessary to fulfill the requirements.
    #     2.  **Identify Entities and Attributes:** Extract distinct entities and their relevant attributes from the user story.
    #     3.  **Determine Relationships:** Identify the relationships between entities (one-to-one, one-to-many, many-to-many).
    #     4.  **Normalize to 3NF:** Design the schema in 3rd Normal Form (3NF) by:
    #         * Eliminating redundant data.
    #         * Ensuring that all non-key attributes are fully dependent on the primary key.
    #         * Removing transitive dependencies.
    #     5.  **Minimize Columns:** Strive to create tables with a reasonable number of columns, breaking down large entities into smaller, related tables where appropriate.
    #     6.  **Define Primary and Foreign Keys:** Clearly define primary keys for each table and establish foreign key relationships to enforce referential integrity.
    #     7.  **Data Types:** Select appropriate data types for each attribute based on the nature of the data.
    #     8.  **Output Format:** Provide the schema in YAML format, clearly defining tables, columns, data types, primary keys, and foreign keys.
    #     9.  **No Explanatory Text:** Do *not* include any explanatory text or comments outside of the YAML schema definition itself.
    #     10. **Assume Best Practices:** Adhere to database design best practices, including proper naming conventions and data type selection.
    #     11. **Focus on Data Structure:** Focus solely on the schema structure, without considering specific database engine syntax or implementation details unless absolutely necessary to define a datatype.
    #     12. **Handle Many-to-Many:** Properly implement many-to-many relationships using junction tables.
        
    #     **Context**

    #     {context}  <-- Previous context

    #     **Input:**

    #     1. **Previously produced Schema that you need to build upon:**
    #     This schema may have incomplete information and may need to be updated based on the new user stories.
    #     {previous_yaml_schema}

    #     2. **User Story and Gherkin:**
    #     {user_story}

    #     **Expected Output:**
    #     1. A schema in YAML format.
        
    #     Example Schema that your final table may look like, basically use it as reference for the structure for the YAML file, do not copy it as it is or replicate it:
    #     ```yaml
    #     tables:
    #         users:
    #             columns:
    #             - id:
    #                 type: Integer
    #                 primary_key: true
    #             - name:
    #                 type: String(100)
    #             - email:
    #                 type: String(150)
    #                 unique: true
    #             relationships:
    #             - user_teams:
    #                 type: one_to_many
    #                 related_table: user_teams
    #                 foreign_key: user_id
    #             - tasks:
    #                 type: one_to_many
    #                 related_table: tasks
    #                 foreign_key: assigned_to
    #             prepopulate:
    #             - name: "John Doe"
    #                 email: "john.doe@example.com"

    #         teams:
    #             columns:
    #             - id:
    #                 type: Integer
    #                 primary_key: true
    #             - name:
    #                 type: String(100)

    #             relationships:
    #             - user_teams:
    #                 type: one_to_many
    #                 related_table: user_teams
    #                 foreign_key: team_id
    #             - projects:
    #                 type: one_to_many
    #                 related_table: projects
    #                 foreign_key: team_id
    #             prepopulate:
    #             - name: "Development Team"

    #         user_teams:
    #             columns:
    #             - id:
    #                 type: Integer
    #                 primary_key: true
    #             - user_id:
    #                 type: Integer
    #                 foreign_key: users.id
    #             - team_id:
    #                 type: Integer
    #                 foreign_key: teams.id
    #             relationships:
    #             - user:
    #                 type: many_to_one
    #                 related_table: users
    #                 foreign_key: user_id
    #             - team:
    #                 type: many_to_one
    #                 related_table: teams
    #                 foreign_key: team_id

    #         projects:
    #             columns:
    #             - id:
    #                 type: Integer
    #                 primary_key: true
    #             - name:
    #                 type: String(150)
    #             - team_id:
    #                 type: Integer
    #                 foreign_key: teams.id
    #             relationships:
    #             - team:
    #               type: many_to_one
    #               related_table: teams
    #                 foreign_key: team_id
    #             - tasks:
    #                 type: one_to_many
    #                 related_table: tasks
    #                 foreign_key: project_id
    #             prepopulate:
    #             - name: "Sample Project"

    #         tasks:
    #             columns:
    #             - id:
    #                 type: Integer
    #                 primary_key: true
    #             - title:
    #                 type: String(150)
    #             - description:
    #                 type: Text
    #             - status:
    #                 type: String(50)
    #             - priority:
    #                 type: String(50)
    #             - due_date:
    #                 type: Date
    #             - project_id:
    #                 type: Integer
    #                 foreign_key: projects.id
    #             - assigned_to:
    #                 type: Integer
    #                 foreign_key: users.id
    #             relationships:
    #             - project:
    #                 type: many_to_one
    #                 related_table: projects
    #                 foreign_key: project_id
    #             - assigned_user:
    #                 type: many_to_one
    #                 related_table: users
    #                 foreign_key: assigned_to
    #             prepopulate:
    #             - title: "Initial Task"
    #               status: "Pending"
    #               priority: "Medium"
    #               due_date: "2025-01-01"
    #     """

        prompt_schema_creation = f"""
        ## Schema Generation Prompt (YAML)

        You are an experienced database schema designer. Design a **practical, optimized, and clean** YAML schema based on the provided user story and context.

        **Key Objectives:**

        1️ **Understand the User Story:**  
           Extract relevant entities, attributes, and relationships from the user story.

        2️ **Design Smart Entities:**  
           Identify entities and prioritize practicality — avoid over-fragmentation unless necessary.

        3️ **Relationships:**  
           - Use junction tables **only** when the many-to-many relationship is complex or needs additional data.  
           - For simple relationships, prioritize **direct foreign keys**.  

        4️ **Normalization (3NF)**:  
           Stick to 3NF, **but don’t overdo it**. If a table gets too fragmented or introduces too many unnecessary joins, intelligently **merge tables**.  

        5 **Optimize the Schema:**  
           - Avoid redundant junction tables unless justified.  
           - Merge tables that hold closely related data to improve readability and performance without causing redundancy.  

        6 **Output Format:**  
           - Deliver only the schema in clean YAML format (no explanations).  
           - Define tables, columns, data types, primary keys, and foreign keys.  

        7️ **Assume Best Practices:**  
           - Use clean, consistent naming conventions.  
           - Select sensible data types without database-specific syntax.  

        **Context:**  
        {context}

        **Previously produced Schema that you need to build upon:**
         This schema may have incomplete information and may need to be updated based on the new user stories.
         {previous_yaml_schema}

         **User Story and Gherkin:**
         {user_story}

        **Expected Output:**  
        YAML schema output (optimized for readability and practicality).
        your yaml format should look like this (do not copy this as it is, use it as reference): 

        ```yaml
         tables:
             users:
                 columns:
                - id:
                    type: Integer
                    primary_key: true
                - name:
                    type: String(100)
                - email:
                    type: String(150)
                    unique: true
                relationships:
                - user_teams:
                    type: one_to_many
                    related_table: user_teams
                    foreign_key: user_id
                - tasks:
                    type: one_to_many
                    related_table: tasks
                    foreign_key: assigned_to
                prepopulate:
                - name: "John Doe"
                    email: "john.doe@example.com"

            teams:
                columns:
                - id:
                    type: Integer
                    primary_key: true
                - name:
                    type: String(100)

                relationships:
                - user_teams:
                    type: one_to_many
                    related_table: user_teams
                    foreign_key: team_id
                - projects:
                    type: one_to_many
                    related_table: projects
                    foreign_key: team_id
                prepopulate:
                - name: "Development Team"

            user_teams:
                columns:
                - id:
                    type: Integer
                    primary_key: true
                - user_id:
                    type: Integer
                    foreign_key: users.id
                - team_id:
                    type: Integer
                    foreign_key: teams.id
                relationships:
                - user:
                    type: many_to_one
                    related_table: users
                    foreign_key: user_id
                - team:
                    type: many_to_one
                    related_table: teams
                    foreign_key: team_id

            projects:
                columns:
                - id:
                    type: Integer
                    primary_key: true
                - name:
                    type: String(150)
                - team_id:
                    type: Integer
                    foreign_key: teams.id
                relationships:
                - team:
                  type: many_to_one
                  related_table: teams
                    foreign_key: team_id
                - tasks:
                    type: one_to_many
                    related_table: tasks
                    foreign_key: project_id
                 prepopulate:
                 - name: "Sample Project"

             tasks:
                 columns:
                 - id:
                     type: Integer
                    primary_key: true
                - title:
                    type: String(150)
                - description:
                    type: Text
                - status:
                    type: String(50)
                - priority:
                    type: String(50)
                - due_date:
                    type: Date
                - project_id:
                    type: Integer
                    foreign_key: projects.id
                - assigned_to:
                    type: Integer
                    foreign_key: users.id
                relationships:
                - project:
                    type: many_to_one
                    related_table: projects
                    foreign_key: project_id
                - assigned_user:
                    type: many_to_one
                    related_table: users
                    foreign_key: assigned_to
                prepopulate:
                - title: "Initial Task"
                  status: "Pending"
                  priority: "Medium"
                  due_date: "2025-01-01"
        """

        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=[system_prompt, prompt_schema_creation],
        #     config={
        #         "temperature": 0.3,  # 0 to 2
        #         "top_p": 0.9,
        #         "top_k": 50,
        #     },
        )
        
        return (
            response.text,
            response.usage_metadata.candidates_token_count,
        )

    except Exception as e:
        st.error(f"Error generating API code: {e}")
        return None
 

def complete_code(past_chat):
    """Completes the unfinshed code based on the past chat."""
    try:
        prompt_api_creation = f"""
        You are an AI completing unfinished code. Below is the conversation history and the incomplete code. Your task is to continue the code from where it was left off without repeating any previous content.

        Instructions:

        1. Do not repeat any part of the already generated code.
        2. Do not add explanations or comments—only output the remaining unfinished code.
        3. Maintain the original style and structure of the code.
        
        Conversation History & Incomplete Code:
        {past_chat}

        Now, continue the code from where it left off. Output only the remaining unfinished code and nothing else.
        """

        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_api_creation,
        )

        return response.text, response.usage_metadata.candidates_token_count

    except Exception as e:
        st.error(f"Error generating API code: {e}")
        return None


def main():
    st.title("Takim User Story Creator")
    st.write("Upload a PDF to receive user stories using GPT-4V.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    user_stories = []
    user_story_count = 0
    api_code = ""

    if uploaded_file:
        # Extract images from PDF
        images = extract_images_from_pdf(uploaded_file.read())

        if images:

            for image in images:  # Iterate through the list of PIL Image objects
                st.image(image, caption=f"Extracted Image", use_column_width=True)

                analysis_result = analyze_image_with_gpt4v(image)

                if analysis_result:
                    st.subheader("Extracted User Stories")
                    st.write(analysis_result)

                    current_user_story = analysis_result  # Store user story for generate_api_code Function
                    user_stories.append(current_user_story)
                    user_story_count += 1
                
                else:
                    st.write("Failed to analyze the image.")




            #schema generation
            st.subheader("Generated YAML Schema")
            combined_user_stories = ""
            yaml_generated = ""

            for i in range(0, user_story_count):
                 combined_user_stories += "\n" + user_stories[i]
                 yaml_generated, token_count = generate_yaml_schema(user_stories[i], combined_user_stories, yaml_generated)
            st.code(yaml_generated, language="yaml")
            


            # Generate boilerplate code based on extracted user stories
            st.subheader("Generated API Code")
            programming_language = st.selectbox(
                "Programming Language",
                ["Python", "JavaScript", "Java", "C#", "Go", "Typescript"],
            )
            database = st.selectbox(
                "Database",
                ["PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server","JSON"],
            )
            orms = st.selectbox("ORM", ["SQLAlchemy", "Sequelize", "Prisma"])
            framework = st.text_input("Preferred API Framework", "Flask")
            additional_instructions = st.text_area(
                "Additional Instructions (Optional)", ""
            )
        

            combined_user_stories = ""
            combined_answer = ""
            chat_number = 1

            if st.button("Generate API Code"):
                boilerplate_code, prompt_boilerplate = generate_boilerplate(
                    analysis_result,
                    programming_language,
                    framework,
                    additional_instructions,
                    database,
                    orms,
                    yaml_generated
                )  # should we be pasing this everytime???

                combined_answer += (
                    "\n\n Prompt "
                    + str(chat_number)
                    + ". #### Prompt for API Boilerplate Code Generation:\n\n"
                    + prompt_boilerplate
                    + "\n\n Answer "
                    + str(chat_number)
                    + ". #### Generated Boilerplate Code from the above prompt:\n\n"
                    + boilerplate_code
                )

                chat_number += 1

                if boilerplate_code:
                    for i in range(0, user_story_count):
                        # st.code(boilerplate_code, language=programming_language.lower())
                        combined_user_stories += "\n" + user_stories[i]

                        if i == 0:

                            api_code, token_count, prompt_api_code = generate_api_code(
                                current_user_story,
                                boilerplate_code,
                                programming_language,
                                framework,
                                additional_instructions,
                                combined_user_stories,
                                database,
                                orms,
                                yaml_generated
                            )

                            combined_answer += (
                                "\n\n Prompt "
                                + str(chat_number)
                                + ". #### Prompt for API Code Generation:\n\n"
                                + prompt_api_code
                                + "\n\n Answer "
                                + str(chat_number)
                                + ". #### Generated API Code from the above prompt:\n\n"
                                + api_code
                            )

                            chat_number += 1

                            while token_count > 8000:
                                print("Alert")
                                remaining_code, token_count = complete_code(combined_answer)

                                api_code += remaining_code
                                combined_answer += remaining_code

                                print(remaining_code)

                        else:
                            api_code, token_count, prompt_api_code = generate_api_code(
                                current_user_story,
                                api_code,
                                programming_language,
                                framework,
                                additional_instructions,
                                combined_user_stories,
                                database,
                                orms,
                                yaml_generated
                            )

                            combined_answer += (
                                "\n\n Prompt "
                                + str(chat_number)
                                + ". #### Prompt for API Code Generation:\n\n"
                                + prompt_api_code
                                + "\n\n Answer "
                                + str(chat_number)
                                + ". #### Generated API Code from the above prompt:\n\n"
                                + api_code
                            )

                            chat_number += 1

                            while token_count > 8000:
                                print("Alert")
                                remaining_code, token_count = complete_code(combined_answer)

                                api_code += remaining_code
                                combined_answer += remaining_code

                                print(remaining_code)

                        # please erase this later
                        x = "Code for iteration: " + str(i)
                        st.write(x)
                        st.code(api_code, language=programming_language.lower())

                    if api_code:
                        st.write("This is the final code")
                        st.code(api_code, language=programming_language.lower())
                else:
                    st.write("Failed to generate Boilerplate code.")

        else:
            st.write("No images were extracted from the PDF.")


if __name__ == "__main__":
    main()
