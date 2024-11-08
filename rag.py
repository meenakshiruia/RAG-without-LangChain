import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from typing import List
from pydantic import BaseModel
import json

# Pydantic models for structured output
class Document(BaseModel):
    document_id: str
    content: str

class RetrievalResult(BaseModel):
    query: str
    retrieved_documents: List[Document]
    retrieval_method: str = "vector search"

class RAGResponse(BaseModel):
    retrieval_result: RetrievalResult
    generated_text: str

# Knowledge base
KNOWLEDGE_BASE = [
    {"id": "doc1", "text": "Python is a high-level programming language known for its simplicity."},
    {"id": "doc2", "text": "Machine Learning is a subset of AI that enables systems to learn from data."},
    {"id": "doc3", "text": "Data structures are ways of organizing and storing data efficiently."},
    {"id": "doc4", "text": "APIs allow different software applications to communicate with each other."},
    {"id": "doc5", "text": "Database management systems are used to store and retrieve data."}
]

# Initialize models
print("Initializing models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Create FAISS index
texts = [doc['text'] for doc in KNOWLEDGE_BASE]
embeddings = embedding_model.encode(texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))

def retrieve_document(query: str) -> Document:
    query_vector = embedding_model.encode([query])[0]
    _, indices = index.search(np.array([query_vector]).astype('float32'), k=1)
    doc_index = indices[0][0]
    relevant_doc = KNOWLEDGE_BASE[doc_index]
    return Document(document_id=relevant_doc['id'], content=relevant_doc['text'])

def generate_answer(query: str, context: str) -> str:
    response = qa_model(question=query, context=context)
    return response['answer']

def rag_query(user_query: str) -> RAGResponse:
    retrieved_doc = retrieve_document(user_query)
    generated_text = generate_answer(user_query, retrieved_doc.content)
    
    return RAGResponse(
        retrieval_result=RetrievalResult(
            query=user_query,
            retrieved_documents=[retrieved_doc]
        ),
        generated_text=generated_text
    )

def main():
    print("RAG system ready!")

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            print("Please enter a valid query.")
            continue

        try:
            result = rag_query(query)
            # Use json.dumps() instead of result.json()
            print(json.dumps(result.model_dump(), indent=2))
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()