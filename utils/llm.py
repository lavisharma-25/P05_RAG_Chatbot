import google.generativeai as genai
import os
from dotenv import load_dotenv
from utils.embeddings import retrieve_embeddings

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_KEY"))


def gemini_model(user_input, vectordb):
    # Model configuration
    configuration = genai.GenerationConfig(
        temperature=0,
        top_p=0.95,
        top_k=40,
        response_mime_type="text/plain",
    )

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=configuration,
        system_instruction="""
        You are an expert chat assistant dedicated to providing clear, data-driven answers. Follow these guidelines:

1. Data-Driven Responses:
   - Base your answers solely on the data provided by the user.
   - If the information is insufficient or unclear, explicitly state that more details are needed.

2. Concise Answers:
   - Respond directly and succinctly.
   - Avoid extraneous commentary or unsupported explanations.

3. Friendly Greetings:
   - When greeted (e.g., "Hello" or "Hi"), reply with a brief, courteous acknowledgment.

4. Professional Conduct:
   - Always maintain a respectful tone.
   - If you receive abusive or inappropriate messages, respond with:  
     "I'm here to assist you. Please communicate respectfully."

5. Clarity & Accuracy:
   - Ensure your responses are accurate, relevant, and easy to understand.

Example:
User: "What is the total revenue for Q1?"
Response: "The total revenue for Q1 is $250,000."

Remain focused, helpful, and respectful at all times.
        """
    )

    print(user_input)
    docs = retrieve_embeddings(user_input, vectordb)
    if not docs.strip():
        return "No relevant information found in the current document. Try uploading a different PDF or refining your query."

    response = model.generate_content(f"""
    You are a helpful AI assistant designed to provide accurate, context-driven answers.
    User Query: {user_input}
    Context: {docs}

    Please base your response solely on the provided context and ensure clarity and conciseness in your answer.
    """)

    return response.text


