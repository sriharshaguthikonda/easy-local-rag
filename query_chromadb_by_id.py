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
            logging.info("Document found. Displaying details:")
            pprint.pprint(result)
        else:
            logging.info("No document found with the specified ID.")
            print("No document found with the specified ID.")
    except Exception as e:
        logging.error(f"An error occurred while retrieving the document: {e}")
        print(f"An error occurred while retrieving the document: {e}")


ids_to_search = [
    "db14ec40d7447e5e036d3ec9c12cdb15642e8b18a6e82ef80c51535a203b0a2f",
    "6410883e30222f4c9c8e36c7827d0c115ef07aa8ab9550dba3bdff1d0bb2a5ee",
    "a7ebaf11023f7e3cc7ba0364d14d0a7f3bcfe88ad6d7e06f065a6280cb0ac14f",
    "67c90fefc4c3b9d9c39a9789b7bffaee544f0c1a385663b3212a812375822f39",
    "8dfc5eef06ce8f70ca14746a4da79d4dd25954b86e66a8c9e08517548acf3ddb",
    "935e01b92791a89906aa85e3ed458a13a0a1f44fbac400f3b227be7463d944a5",
    "168b51d67497caad157d0a93e583cedad6b311a6acd962bef1cc7f42cfe4960b",
    "a8365d28d1b77085b7bffbf93dd6ca01bf0e1f1d2edcec9ead1909a50655f85b",
    "5086d97d7eb8c913b55342d464ca04be8e6d12f436cec6a355aed184d97e5c9c",
    "e334f7825f2288f8ffe4fb6f48d997604b9f359a0a631027ee75e027d713ef0c",
    "61e1a56d4ea0116b8927c7a6a2c1af2e9598b3ea983d6d84bbc8d7ffdef0f91b",
    "2be57fbb9cc40384edee59b8fd8668341e6b817cd7e2f3ab35871684476c8e35",
    "b2821bd7fddce906349f3aa06507d982d8fedabfe5abfddecfc2a2bc02bc8f47",
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
    # search_by_id(collection, "fa70396621a2114acb2c818e2f55ddf78740e2373e5ed8bcbe32dfed0f42a0c0")
    search_by_ids(collection, ids_to_search)
