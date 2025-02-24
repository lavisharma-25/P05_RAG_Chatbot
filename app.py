import streamlit as st
import os
from dotenv import load_dotenv
from utils.embeddings import create_and_store_embeddings, delete_embeddings
from utils.llm import gemini_model

load_dotenv()  # Load environment variables from .env file

def main():
    st.title("Document Based QA App")

    # Ensure the Data directory exists
    os.makedirs("Data", exist_ok=True)

    # Initialize session state for vector_db and current PDF
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None

    # File uploader for PDF
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf is not None:
        # Check if a new PDF is uploaded
        if st.session_state.current_pdf != uploaded_pdf.name:
            # st.info("New PDF detected. Clearing previous embeddings...")

            # Clear previous embeddings and reset session
            if st.session_state.vector_db is not None:
                delete_embeddings(st.session_state.vector_db)
                st.session_state.vector_db = None

            # Clear Streamlit cache to avoid stale data
            st.cache_data.clear()
            st.cache_resource.clear()
            # st.info("Previous embeddings and cache cleared.")

            # Save the uploaded PDF to Data directory
            pdf_path = os.path.join("Data", uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            # st.success(f"PDF uploaded successfully to: {pdf_path}")

            # Create new embeddings and store in the database
            with st.spinner("Creating embeddings, please wait..."):
                st.session_state.vector_db = create_and_store_embeddings(pdf_path)
                st.session_state.current_pdf = uploaded_pdf.name

                # if st.session_state.vector_db:
                #     st.success("Embeddings created successfully!")
                # else:
                #     st.error("Failed to create embeddings. Please check the PDF content or embedding function.")

    # Text input for user query
    user_query = st.text_input("Enter your query:")

    # Button to get the answer
    if st.button("Retrieve"):
        if st.session_state.vector_db is None:
            st.warning("Please upload a PDF and create embeddings first.")
        else:
            with st.spinner("Retrieving answer..."):
                response = gemini_model(user_query, st.session_state.vector_db)
                if response:
                    st.write(f"**Answer for {st.session_state.current_pdf}:**")
                    st.write(response)
                else:
                    st.warning("No response found. Try another query or re-upload the PDF.")

if __name__ == "__main__":
    main()
