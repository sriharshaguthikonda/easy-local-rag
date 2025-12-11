import chromadb

def list_collections():
    try:
        # Initialize the Chroma client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # List all collections
        collections = client.list_collections()
        
        if not collections:
            print("No collections found in ChromaDB.")
            return
            
        print("\nAvailable collections in ChromaDB:")
        print("-" * 50)
        for i, collection in enumerate(collections, 1):
            print(f"{i}. Name: {collection.name}")
            print(f"   Metadata: {collection.metadata}")
            print(f"   Count: {collection.count()}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    list_collections()
