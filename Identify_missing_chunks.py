import chromadb
from chromadb.config import Settings
import logging
from difflib import SequenceMatcher, ndiff
import nltk
import pprint

from Semantic_chunking import (
    extract_text_from_html,
    split_into_chunks,
    generate_chunk_id,
)


# Configure logging with colors
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;21m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
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


def get_chunks_from_chromadb(file_path):
    """Retrieve all chunks for a specific file from ChromaDB."""
    client = chromadb.PersistentClient(settings=Settings(persist_directory="./chroma"))
    collection = client.get_or_create_collection(name="html_chunks_temp")

    results = collection.get(
        where={"file_name": file_path},
        include=["documents", "embeddings", "metadatas", "uris"],
    )

    chunks = {
        results["ids"][i]: results["metadatas"][i]["text"]
        for i in range(len(results["ids"]))
    }
    logging.info(f"\033[92mChunks retrieved from ChromaDB:\033[0m {chunks}")
    return chunks


def print_chunks(chunks):
    """Print chunk IDs and texts."""
    for chunk_id, text in chunks.items():
        print(f"\033[94m{chunk_id} === {text}\033[0m")


def calculate_chunk_similarity(chunk1, chunk2):
    """Calculate similarity between two chunks using SequenceMatcher."""
    return SequenceMatcher(None, chunk1, chunk2).ratio()


def align_chunks(new_chunks, chromadb_chunks, similarity_threshold=0.99):
    """Align new chunks with existing chunks and print the alignment."""
    alignments = []
    non_aligned_new_chunks = []
    non_aligned_chromadb_chunks = set(chromadb_chunks.keys())

    for new_chunk in new_chunks:
        best_match = None
        best_similarity = 0
        for existing_chunk_id, existing_chunk_text in chromadb_chunks.items():
            similarity = calculate_chunk_similarity(
                new_chunk["text"], existing_chunk_text
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (existing_chunk_id, existing_chunk_text)
        if best_similarity >= similarity_threshold:
            alignments.append((new_chunk, best_match, best_similarity))
            non_aligned_chromadb_chunks.discard(best_match[0])
        else:
            non_aligned_new_chunks.append(new_chunk)

    return alignments, non_aligned_new_chunks, non_aligned_chromadb_chunks


def print_alignments(alignments):
    """Print the alignments between new chunks and existing chunks."""
    for new_chunk, best_match, similarity in alignments:
        print(f"\033[94mNew Chunk: {new_chunk['text']}\033[0m")
        print(f"\033[94mBest Match: {best_match[1]}\033[0m")
        print(f"\033[94mSimilarity: {similarity}\033[0m")
        print("-" * 80)


def check_partial_alignment(non_aligned_chunks, text, similarity_threshold=0.5):
    """Check if non-aligned chunks align with some part of the text."""
    partial_alignments = []
    for chunk in non_aligned_chunks:
        best_match = None
        best_similarity = 0
        for sentence in text.split("."):
            similarity = calculate_chunk_similarity(chunk["text"], sentence)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = sentence
        if best_similarity >= similarity_threshold:
            partial_alignments.append((chunk, best_match, best_similarity))
    return partial_alignments


def print_partial_alignments(partial_alignments):
    """Print the partial alignments between non-aligned chunks and text."""
    for chunk, best_match, similarity in partial_alignments:
        print(f"\033[93mNon-aligned Chunk: {chunk['text']}\033[0m")
        print(f"\033[93mPartial Match: {best_match}\033[0m")
        print(f"\033[93mSimilarity: {similarity}\033[0m")
        print("-" * 80)


def print_non_aligned_chunks(
    non_aligned_new_chunks, non_aligned_chromadb_chunks, chromadb_chunks
):
    """Print the non-aligned new and existing chunks."""
    print("\033[91mNon-aligned new chunks:\033[0m")
    for chunk in non_aligned_new_chunks:
        print(f"\033[91m{chunk['id']} === {chunk['text']}\033[0m")
        print("-" * 80)

    print("\033[91mNon-aligned existing chunks:\033[0m")
    for chunk_id in non_aligned_chromadb_chunks:
        print(f"\033[91m{chunk_id} === {chromadb_chunks[chunk_id]}\033[0m")
        print("-" * 80)


def reconstruct_text_from_chunks(chunks):
    """Reconstruct text from chunks using the overlapped regions."""
    sorted_chunks = sorted(chunks.items(), key=lambda x: x[0])
    reconstructed_text = sorted_chunks[0][1]
    for i in range(1, len(sorted_chunks)):
        overlap = SequenceMatcher(
            None, reconstructed_text, sorted_chunks[i][1]
        ).find_longest_match(0, len(reconstructed_text), 0, len(sorted_chunks[i][1]))
        if overlap.size > 20:
            reconstructed_text += sorted_chunks[i][1][overlap.b + overlap.size :]
        else:
            reconstructed_text += sorted_chunks[i][1]
    return reconstructed_text


def print_diff(text1, text2):
    """Print the differences between two texts using colors."""
    diff = ndiff(text1.splitlines(), text2.splitlines())
    for line in diff:
        if line.startswith("+"):
            print(f"\033[92m{line}\033[0m")  # Green for additions
        elif line.startswith("-"):
            print(f"\033[91m{line}\033[0m")  # Red for deletions
        else:
            print(line)


def check_chromadb_chunks_in_text(chromadb_chunks, text):
    """Check if chromadb_chunks are present in text_extracted_from_html."""
    missing_chunks = []
    for chunk_id, chunk_text in chromadb_chunks.items():
        if chunk_text not in text:
            # Split the chunk into sentences
            sentences = nltk.sent_tokenize(chunk_text)
            all_sentences_present = all(sentence in text for sentence in sentences)
            if not all_sentences_present:
                missing_chunks.append((chunk_id, chunk_text))
    return missing_chunks


def print_missing_chunks(missing_chunks):
    """Print the missing chunks that are not found in the text."""
    print(
        "\033[91mMissing chunks from ChromaDB that are not found in the extracted text:\033[0m"
    )
    for chunk_id, chunk_text in missing_chunks:
        print(f"\033[91m{chunk_id} === {chunk_text}\033[0m")
        print("-" * 80)


if __name__ == "__main__":
    file_path = r"C:\Users\deletable\Downloads\deletable\Cummings Otolaryngology\100_Diagnostic_Imaging_of_the_Pharynx_and_Esophagus.html"
    logging.info(f"\033[92mRetrieving chunks for file: {file_path}\033[0m")
    chromadb_chunks = get_chunks_from_chromadb(file_path)
    logging.info(f"\033[92mFound {len(chromadb_chunks)} chunks in ChromaDB.\033[0m")

    reconstructed_text = reconstruct_text_from_chunks(chromadb_chunks)
    logging.info("\033[92mReconstructed text from ChromaDB chunks.\033[0m")
    print("\033[92mReconstructed text from ChromaDB:\033[0m")
    print(reconstructed_text)

    text_extracted_from_html = extract_text_from_html(file_path)
    logging.info("\033[92mExtracted text from the file.\033[0m")

    text_extracted_from_html_chunks = split_into_chunks(text_extracted_from_html)
    logging.info(
        f"\033[92mSplit text into {len(text_extracted_from_html_chunks)} new chunks.\033[0m"
    )

    text_extracted_from_html_chunks_dict = {
        chunk["id"]: chunk["text"] for chunk in text_extracted_from_html_chunks
    }

    #        remove chromadb_chunks[id] if ids match if those in  text_extracted_from_html_chunks_dict
    chromadb_chunks_with_missing_ids = {
        id: text
        for id, text in chromadb_chunks.items()
        if id not in text_extracted_from_html_chunks_dict
    }

    print(
        "\033[92m number of chromadb_chunks with missing ids:\033[0m",
        len(chromadb_chunks),
    )

    # pprint.pprint(  chromadb_chunks_with_missing_ids,)
    print(
        "\033[92m number of chromadb_chunks_with_missing_ids with missing ids:\033[0m",
        len(chromadb_chunks_with_missing_ids),
    )

    missing_chromadb_chunks = check_chromadb_chunks_in_text(
        chromadb_chunks, text_extracted_from_html
    )
    logging.info(
        f"\033[91mFound {len(missing_chromadb_chunks)} missing_chromadb_chunks in the text_extracted_from_html.\033[0m"
    )

    print_missing_chunks(missing_chromadb_chunks)

    #        remove chromadb_chunks[id] if ids match if those in  text_extracted_from_html_chunks_dict
    text_extracted_from_html_chunks_with_missing_ids = {
        id: text
        for id, text in text_extracted_from_html_chunks_dict.items()
        if id not in chromadb_chunks
    }

    print(
        "\033[92m number of chromadb_chunks with missing ids:\033[0m",
        len(text_extracted_from_html_chunks_dict),
    )

    # pprint.pprint(  chromadb_chunks_with_missing_ids,)
    print(
        "\033[92m number of chromadb_chunks_with_missing_ids with missing ids:\033[0m",
        len(text_extracted_from_html_chunks_with_missing_ids),
    )

    missing_text_extracted_from_html_chunks = check_chromadb_chunks_in_text(
        text_extracted_from_html_chunks_dict, reconstructed_text
    )
    logging.info(
        f"\033[91mFound {len(missing_text_extracted_from_html_chunks)} missing chunks in the extracted text.\033[0m"
    )

    print_missing_chunks(missing_text_extracted_from_html_chunks)

    logging.info(
        "\033[92mPrinting differences between reconstructed text and extracted text.\033[0m"
    )
    # print_diff(reconstructed_text, text_extracted_from_html)

    # Beep to indicate the script has ended
    import winsound

    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
