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


def list_databases():
    logging.info("Initializing ChromaDB client with the specified database path.")
    # Initialize ChromaDB client with the specified database path
    client = chromadb.PersistentClient(settings=Settings(persist_directory="./chroma"))

    logging.info("Getting or creating the collection 'html_chunks_temp'.")
    # Get or create the collection
    collection = client.get_or_create_collection(name="html_chunks_temp")

    if collection:
        logging.info("Collection 'html_chunks_temp' found.")
        print("Databases present in ChromaDB:")
        results = collection.get(
            where={
                # "file_name": r"C:\Users\deletable\Google Drive\goodman Gilman 12e\II. Neuropharmacology\24. Drug Addiction.htm"
                "file_name": r"C:\Users\deletable\Downloads\deletable\Cummings Otolaryngology\100_Diagnostic_Imaging_of_the_Pharynx_and_Esophagus.html"
            },
            include=[
                "documents",
                "embeddings",
                "metadatas",
                "uris",
            ],
        )
        logging.info("Query executed. Printing the results.")
        # Print the entire structure of the results
        print(type(results))
        logging.info(" pulling ids from the results.")
        print((results["ids"]))
        print((results["metadatas"][1]["file_name"]))
        print(len(results["metadatas"][1]["file_name"]))

        logging.info(" pulling embeddings  from the results.")
        print((results["embeddings"]))

        logging.info(" pulling  embeddings subdetails.")
        # print((results["embeddings"][1]))
        # pprint.pprint(results)
        pprint.pprint(results.keys())
        if "metadatas" in results:
            # pprint.pprint(results["metadatas"])
            pprint.pprint(results["metadatas"][1])
        else:
            logging.info("No documents found in this collection.")
            print("No documents found in this collection.")

        logging.info(f"Number of metadatas found: {len(results['metadatas'])}")
        print(len(results["metadatas"]))

    else:
        logging.info("No collection found.")
        print("No databases found in ChromaDB.")


if __name__ == "__main__":
    logging.info("Starting the script to list databases in ChromaDB.")
    list_databases()
    logging.info("Script execution completed.")
