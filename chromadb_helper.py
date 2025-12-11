"""
Helper script to run ChromaDB queries in isolation from Qt.
Called via subprocess to avoid Qt/ChromaDB conflicts.
"""
import sys
import json
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import ollama
from rank_bm25 import BM25Okapi
import re

def query_chromadb(chromadb_path, collection_name, embedding, n_results=20):
    """Query ChromaDB and return results"""
    client = chromadb.PersistentClient(
        path=chromadb_path,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_collection(collection_name)
    
    result = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return result

def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    chromadb_path = input_data['chromadb_path']
    collection_name = input_data['collection_name']
    embedding = input_data['embedding']
    n_results = input_data.get('n_results', 20)
    
    try:
        result = query_chromadb(chromadb_path, collection_name, embedding, n_results)
        
        # Convert to JSON-serializable format
        output = {
            'success': True,
            'documents': result['documents'][0] if result.get('documents') else [],
            'metadatas': result['metadatas'][0] if result.get('metadatas') else [],
            'distances': result['distances'][0] if result.get('distances') else [],
        }
    except Exception as e:
        output = {
            'success': False,
            'error': str(e)
        }
    
    print(json.dumps(output))

if __name__ == '__main__':
    main()
