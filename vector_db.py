import os
import chromadb
import embeddings

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "corpus_chunks"

def get_chroma_client():
    """Initializes and returns the ChromaDB persistent client."""
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

def get_collection():
    """Gets or creates the ChromaDB collection with cosine distance metric."""
    client = get_chroma_client()
    # We specify cosine distance metric: hnsw:space = cosine
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def add_document_chunks(doc_id: int, chunks: list[str]) -> int:
    """
    Generates embeddings for chunks of text and adds them to ChromaDB.
    Returns the number of chunks added.
    """
    if not chunks:
        return 0
        
    collection = get_collection()
    
    # Generate embeddings in batch
    vectors = embeddings.get_embeddings_batch(chunks)
    
    ids = [f"chunk_{doc_id}_{idx}" for idx in range(len(chunks))]
    metadatas = [{"corpus_doc_id": doc_id, "chunk_index": idx} for idx in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=vectors,  # type: ignore
        metadatas=metadatas, # type: ignore
        documents=chunks
    )
    
    return len(chunks)

def delete_document_chunks(doc_id: int):
    """Deletes all vector chunks associated with a specific document ID from ChromaDB."""
    collection = get_collection()
    # Delete based on metadata filter
    collection.delete(where={"corpus_doc_id": doc_id})

def search_similar_chunks(
    query_text: str,
    top_k: int = 3,
    similarity_threshold: float = 0.5
) -> list[dict]:
    """
    Searches ChromaDB for the top K closest chunks.
    Filters out results with similarity score below the threshold.
    Returns a list of dicts: {'corpus_doc_id': int, 'chunk_index': int, 'text': str, 'similarity': float}
    """
    if not query_text.strip():
        return []
        
    collection = get_collection()
    query_vector = embeddings.get_embedding(query_text)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Check if we have results
    if not results or not results["ids"] or not results["ids"][0]:
        return []
        
    matches = []
    ids = results["ids"][0]
    if results["distances"]:
        distances = results["distances"][0]
    if results["metadatas"]:
        metadatas = results["metadatas"][0]
    if results["documents"]:
        documents = results["documents"][0]
    
    if not (distances and metadatas and documents):
        return []
    
    for i in range(len(ids)):
        # ChromaDB cosine distance d = 1 - cosine_similarity
        # Therefore similarity = 1.0 - d
        distance = distances[i]
        similarity = 1.0 - distance
        
        if similarity >= similarity_threshold:
            matches.append({
                "corpus_doc_id": metadatas[i]["corpus_doc_id"],
                "chunk_index": metadatas[i]["chunk_index"],
                "text": documents[i],
                "similarity": similarity
            })
            
    # Sort matches by similarity descending
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches

def get_expanded_chunk_context(doc_id: int, chunk_index: int) -> str:
    """
    Fetches the matched chunk along with its immediately preceding and succeeding
    chunks from ChromaDB to reconstruct a wider context for the LLM.
    """
    collection = get_collection()
    
    # We want to fetch chunk_index - 1, chunk_index, chunk_index + 1
    target_ids = []
    if chunk_index > 0:
        target_ids.append(f"chunk_{doc_id}_{chunk_index - 1}")
    target_ids.append(f"chunk_{doc_id}_{chunk_index}")
    target_ids.append(f"chunk_{doc_id}_{chunk_index + 1}")
    
    try:
        results = collection.get(ids=target_ids)
        if results and results["documents"] and results["ids"]:
            # ChromaDB get doesn't guarantee sorting order.
            # Map documents by their ID and reconstruct the sorted list.
            id_to_doc = dict(zip(results["ids"], results["documents"]))
            sorted_docs = []
            for tid in target_ids:
                if tid in id_to_doc:
                    sorted_docs.append(id_to_doc[tid])
            return " ".join(sorted_docs)
    except Exception as e:
        print(f"Error retrieving expanded context: {e}")
        
    return ""

def get_chunk_text(doc_id: int, chunk_index: int) -> str:
    """Fetches a single corpus chunk text from ChromaDB."""
    collection = get_collection()
    chunk_id = f"chunk_{doc_id}_{chunk_index}"

    try:
        results = collection.get(ids=[chunk_id])
        if results and results["documents"]:
            return results["documents"][0] or ""
    except Exception as e:
        print(f"Error retrieving chunk text: {e}")

    return ""
