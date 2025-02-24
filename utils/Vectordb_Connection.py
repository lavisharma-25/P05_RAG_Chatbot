from langchain_postgres.vectorstores import PGVector
import os
import psycopg2
from dotenv import load_dotenv
from uuid import uuid4
import os.path

load_dotenv()

CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING")
COLLECTION_NAME = "AI_COLLECTION_DATABASE"


def get_psycopg2_connection_string(conn_str):
    """
    Convert a SQLAlchemy connection string to one acceptable by psycopg2.
    For example, replaces 'postgresql+psycopg://' with 'postgresql://'.
    """
    if conn_str.startswith("postgresql+psycopg://"):
        return conn_str.replace("postgresql+psycopg://", "postgresql://", 1)
    return conn_str


def collection_exists(collection_name):
    """
    Check if the collection (table) exists in the database.
    This function uses PostgreSQL's to_regclass to check if a table by the given name exists.
    """
    valid_conn_str = get_psycopg2_connection_string(CONNECTION_STRING)
    conn = psycopg2.connect(valid_conn_str)
    cur = conn.cursor()
    cur.execute("SELECT to_regclass(%s);", (collection_name,))
    exists = cur.fetchone()[0] is not None
    cur.close()
    conn.close()
    return exists


def connect_db(collection_name=COLLECTION_NAME, embeddings=None, documents=None, pdf_file_path: str = None):
    """
    Connect to the PGVector database and store documents with embeddings.

    If pdf_file_path is provided, deterministic IDs based on the PDF's file name are created.
    This ensures that if embeddings for the same PDF already exist, they are overwritten
    with the new embeddings instead of adding duplicate entries.

    Args:
        collection_name (str): The name of the collection in the database.
        embeddings: The embedding model instance.
        documents (list): A list of text chunks (documents) to embed.
        pdf_file_path (str, optional): The path to the source PDF file.

    Returns:
        The PGVector database object with stored embeddings.
    """
    if documents is None:
        raise ValueError("Documents must be provided.")

    # If pdf_file_path is provided, create deterministic IDs using the file name
    if pdf_file_path:
        file_identifier = os.path.splitext(os.path.basename(pdf_file_path))[0]
        ids = [f"{file_identifier}_{i}" for i in range(len(documents))]
    else:
        ids = [str(uuid4()) for _ in documents]

    # Check if the collection exists in the database
    if not collection_exists(collection_name):
        # Create a new collection if it does not exist
        db = PGVector.from_documents(
            embedding=embeddings,
            collection_name=collection_name,
            connection=CONNECTION_STRING,
            documents=documents,
            ids=ids
        )
    else:
        # Collection exists, so instantiate the vector store and add documents
        db = PGVector(embeddings=embeddings, collection_name=collection_name, connection=CONNECTION_STRING)
        db.add_documents(documents=documents, ids=ids)

    return db
