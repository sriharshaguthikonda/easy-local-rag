import chromadb
from chromadb.config import Settings
import pprint
import logging


# Configure logging with colors
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Set up logging
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])


def search_by_id(collection, document_id):
    logging.info(f"Searching for document with ID: {document_id}")
    try:
        result = collection.get(
            ids=[document_id], include=["embeddings", "metadatas", "documents"]
        )
        if result and result["documents"]:
            for document in result["documents"]:
                logging.info("Document found. Displaying details:")
                pprint.pprint(document)
            logging.info("Document found. Displaying details:")
            pprint.pprint(result)
        else:
            logging.info("No document found with the specified ID.")
            print("No document found with the specified ID.")
    except Exception as e:
        logging.error(f"An error occurred while retrieving the document: {e}")
        print(f"An error occurred while retrieving the document: {e}")


ids_to_search = [
    "23eadf8e2891189879510866f1e54b8127b5cee2ad3df80640d69a22dff582a5",
    "02d54b84e016bbd2625f0a10721c07c10776135b767fa7fbdd00c369d638e30d",
    "23eadf8e2891189879510866f1e54b8127b5cee2ad3df80640d69a22dff582a5",
]


def search_by_ids(collection, document_ids):
    """
    Retrieve and display documents from the collection based on a list of document IDs.

    Args:
        collection: The ChromaDB collection object.
        document_ids (list): A list of document IDs to retrieve.

    Returns:
        None
    """
    logging.info(f"Searching for documents with IDs: {document_ids}")
    try:
        # Retrieve documents matching the provided IDs
        results = collection.get(
            ids=document_ids, include=["embeddings", "metadatas", "documents"]
        )
        # Check if any documents were found
        if results and results.get("documents"):
            logging.info(
                f"Found {len(results['documents'])} documents. Displaying details:"
            )
            for i in range(len(results["documents"])):
                logging.info(f"Document ID: {results['ids'][i]}")
                logging.info(f"Metadata: {results['metadatas'][i]}")
                logging.info(f"Document: {results['documents'][i]}")
                logging.info(f"Embedding: {results['embeddings'][i]}")
        else:
            logging.info("No documents found with the specified IDs.")
            print("No documents found with the specified IDs.")
    except Exception as e:
        logging.error(f"An error occurred while retrieving the documents: {e}")
        print(f"An error occurred while retrieving the documents: {e}")


if __name__ == "__main__":
    logging.info("Initializing ChromaDB client with the specified database path.")
    client = chromadb.PersistentClient(settings=Settings(persist_directory="./chroma"))

    logging.info("Getting or creating the collection 'html_chunks_temp'.")
    collection = client.get_or_create_collection(name="html_chunks_temp")

    # Replace 'your_document_id' with the actual ID of the document you want to search for
    document_id = "your_document_id"
    # search_by_id(  collection, "4c8fa7b5ee3fa96f6b1ce218d1675d89f70c1cadb83e634a78d029eb31302b5d")
    search_by_ids(collection, ids_to_search)
