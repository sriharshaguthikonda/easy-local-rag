import hashlib
import os
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk

from rag_config import get_optional_path
from vault_store import atomic_write_json, load_vault, merge_vault_entries


def _get_sentence_tokenizer() -> Any:
    try:
        return nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)
        return nltk.data.load("tokenizers/punkt/english.pickle")


def clean_text(text):
    cleaned_text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", cleaned_text).strip() + " "


def extract_text_from_html(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as html_file:
            return clean_text(BeautifulSoup(html_file, "html.parser").get_text(separator=" ", strip=True))
    except Exception as error:
        print(f"Error processing file {file_path}: {error}")
        return ""


def generate_chunk_id(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_into_chunks(text):
    chunks = []
    current_chunk = ""
    for sentence in _get_sentence_tokenizer().tokenize(text.strip()):
        if len(current_chunk) + len(sentence) + 1 < 1000:
            current_chunk += (sentence + " ").strip()
        else:
            if current_chunk:
                chunks.append({"id": generate_chunk_id(current_chunk), "text": current_chunk.strip()})
            current_chunk = (sentence + " ").strip()
    if current_chunk:
        chunks.append({"id": generate_chunk_id(current_chunk), "text": current_chunk.strip()})
    return chunks


def convert_html_to_json(directory_path: str | os.PathLike[str], vault_path: str | os.PathLike[str] = "vault.json") -> None:
    source = Path(directory_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_dir():
        raise NotADirectoryError(source)
    existing_data = load_vault(vault_path)
    html_files = [
        Path(root) / file_name
        for root, _, files in os.walk(source)
        for file_name in files
        if file_name.lower().endswith((".htm", ".html", ".xhtml", ".shtml", ".dhtml"))
    ]
    with Pool(cpu_count()) as pool:
        all_texts = list(tqdm(pool.imap(extract_text_from_html, map(str, html_files)), total=len(html_files), desc="Processing HTML files"))
    new_data = []
    for file_path, text in zip(html_files, all_texts):
        if text:
            new_data.append({
                "file_name": os.path.normpath(str(file_path.resolve())),
                "modification_time": file_path.stat().st_mtime,
                "chunks": split_into_chunks(text),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
    atomic_write_json(vault_path, merge_vault_entries(existing_data, new_data))
    print(f"Vault updated: {len(new_data)} new/updated entries written to {vault_path}.")


def main():
    configured_path = get_optional_path("EASY_RAG_VAULT_SOURCE_DIR")
    if configured_path and configured_path.exists():
        directory_path = str(configured_path)
    else:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        directory_path = filedialog.askdirectory()
    if directory_path:
        convert_html_to_json(directory_path)


if __name__ == "__main__":
    main()
