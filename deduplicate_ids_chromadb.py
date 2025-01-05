import chromadb
from collections import defaultdict


""" TODO not working yet"""


# Initialize ChromaDB client and connect to your collection
client = chromadb.Client()
collection_name = "html_chunks_temp"  # Replace with your collection name
collection = client.get_or_create_collection(collection_name)

# Step 1: Retrieve all documents with their IDs and text content
results = collection.get(include=["documents", "metadatas"])

# Assuming 'results' is obtained from the collection.get() method
all_ids = print(results)

# Extract the 'text' field from each metadata dictionary
all_texts = [metadata["text"] for metadata in results["metadatas"]]

# Print the IDs and corresponding texts
print(all_ids)
print(all_texts)

"""
# Step 2: Identify duplicate texts and their corresponding IDs
text_to_ids = defaultdict(list)
for doc_id, text in zip(all_ids, all_texts):
    text_to_ids[text].append(doc_id)

# Step 3: Determine which IDs to delete (keep one ID per unique text)
ids_to_delete = []
for ids in text_to_ids.values():
    if len(ids) > 1:
        # Keep the first ID and mark the rest for deletion
        ids_to_delete.extend(ids[1:])

# Step 4: Delete duplicate documents from the collection
if ids_to_delete:
    collection.delete(ids=ids_to_delete)
    print(f"Deleted duplicate IDs: {ids_to_delete}")
else:
    print("No duplicate documents found.")"""
