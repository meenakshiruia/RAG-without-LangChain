RAG-without-LangChain

Build a Simple RAG System

Objective: Implement a basic Retrieval-Augmented Generation system that takes a user query, retrieves relevant information from a small knowledge base, and generates a response.

Instructions:

Step 1: Create a small knowledge base (e.g., a collection of short text documents related to a topic of your choice, like famous landmarks, programming concepts, etc.).

Step 2: Use a library like Sentence Transformers to generate vector embeddings for each document in the knowledge base.

Step 3: Set up a retrieval mechanism using a simple vector search (you can use libraries like FAISS or Scikit-Learn for this purpose).

Step 4: Create a function that takes a user query, retrieves relevant documents, and uses a pre-trained language model to generate a response.

Optional : Get output in specific format using pydantics
