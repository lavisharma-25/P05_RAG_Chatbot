import torch
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from utils.pdf_loader import extract_text_from_pdf, divide_text_into_chunks
from utils.Vectordb_Connection import connect_db

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': DEVICE},
    show_progress=True
)

def create_and_store_embeddings(pdf_file_path):
    """
    Extract text from a PDF file, divide it into chunks, generate embeddings,
    and store them in a vector database.

    Args:
        pdf_file_path (str): The file path to the PDF document.

    Returns:
        vector_db: The vector database object with stored embeddings.
    """
    try:
        logger.info(f"Extracting text from PDF: {pdf_file_path}")
        documents = extract_text_from_pdf(pdf_file_path)
        logger.info("Text extraction completed. Dividing text into chunks...")

        chunks = divide_text_into_chunks(documents)
        logger.info(f"Text divided into {len(chunks)} chunks. Generating embeddings...")

        _vector_db = connect_db(embeddings=embedding_model, documents=chunks, pdf_file_path=pdf_file_path)  # Collection Name is optional
        logger.info("Connected to database successfully. Embeddings created and added to database.")

        return _vector_db
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

def retrieve_embeddings(_query: str, db):
    """
    Retrieve the most similar embeddings from the vector database based on a query.

    Args:
        _query (str): The query string to search for.
        db: The vector database object.

    Returns:
        str: A concatenated string of document contents ranked by similarity score.
    """
    try:
        query_embeddings = embedding_model.embed_query(_query)
        similar_docs = db.similarity_search_with_score_by_vector(
            embedding=query_embeddings,
            k=5,
        )
        if not similar_docs:
            logger.warning("No matching documents found in embeddings!")
            return ""
        logger.info(f"Found {len(similar_docs)} similar documents.")
        # Unpack each tuple (doc, score) returned by the search
        return " ".join(_doc.page_content for _doc, _ in similar_docs)
    except Exception as e:
        logger.error(f"An error occurred during retrieval: {e}")
        raise


def delete_embeddings(db, ids: list = None):
    try:
        if ids:
            db.delete(ids=ids)
            logger.info(f"Deleted embeddings with IDs: {ids}")
        else:
            # Clear the entire collection and force DB to reset
            db.delete_collection()
            db = None  # Drop the reference
            logger.info("Deleted all embeddings from the collection and reset DB reference.")
    except Exception as e:
        logger.error(f"An error occurred during deletion: {e}")
        raise


if __name__ == '__main__':
    pdf_path = "../Data/budget_speech.pdf"
    logger.info(f"Starting embedding process for {pdf_path}")
    vector_db = create_and_store_embeddings(pdf_path)
    logger.info("Embedding process completed.")

    # # Optionally, perform a query to retrieve similar documents
    # query = "Tell me about new assessment model for MSME credit"  # Replace with your actual query
    # results = retrieve_embeddings(query, vector_db)
    # for idx, doc in enumerate(results, start=1):
    #     logger.info(f"Document {idx}: {doc}")

    # Example: Deleting embeddings by IDs (or leave ids=None to delete all)
    # ids_to_delete = ["document_id_1", "document_id_2"]  # Replace with actual IDs if available
    # delete_embeddings(vector_db, ids=ids_to_delete)
    # Alternatively, to delete all embeddings:
    delete_embeddings(vector_db)
