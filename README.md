# RAG System

## How to Run

1. Create virtual environment:

    python3 -m venv venv

2. Activate it:
   
    source venv/bin/activate

3. Install dependencies:
   
    pip install -r requirements.txt

4. Create a `.env` file in the project root and add your OpenAI API key:
   
    OPENAI_API_KEY=your_api_key_here

5. Run the app:
    
    python3 RAG_app.py

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions using a selected document. The system retrieves relevant text chunks using embeddings and FAISS, then generates answers using a language model. A reranker is used to improve the relevance of retrieved chunks.

## Selected Document

The document used in this project is a Wikipedia article on:
*Artificial Intelligence in Video Games*
The text was extracted using a custom scraper and saved as Selected_Document.txt.

## How It Works

1. Text Extraction
    - The document is scraped using BeautifulSoup and saved locally.

2. Chunking
    - The document is split into chunks using:
        - Chunk size: 500
        - Overlap: 50

3. Embeddings
    - Each chunk is converted into embeddings using:
        - `sentence-transformers/all-distilroberta-v1`

4. Vector Search (FAISS)
    - Embeddings are stored in a FAISS index.
    - Top 20 most similar chunks are retrieved.

5. Reranking
    - Retrieved chunks are reranked using:
        - cross-encoder/ms-marco-MiniLM-L-6-v2
    - Top 8 chunks are selected after reranking.

6. Answer Generation
    - The model generates answers using only the provided context.
    - If the answer is not found, the system returns:
        - "I don't know. ○|￣|_"

## Example Questions & Results

**Q:** What is this document about?
**A:** The document discusses the impact of generative AI in the video game industry and how AI is used in gameplay and development.

**Q:** What is FEAR about?
**A:** The 2005 psychological horror FPS F.E.A.R involves battling cloned soldiers, robots, and paranormal enemies.

**Q:** When is my birthday?
**A:** I don’t know. ○|￣|_

## Chunking Experiment

I tested different chunk sizes and overlap values to evaluate retrieval quality.

### Test 1:

- Chunk size: 300
- Overlap: 50

Result: More precise retrieval but less context.

### Test 2:

- Chunk size: 500
- Overlap: 50

Result: Balanced performance (chosen configuration).

### Test 3:

- Chunk size: 800
- Overlap: 100

Result: More context but slightly less precise retrieval.

## Deep Dive Questions

1. How does AI contribute to gameplay in video games?
AI controls NPC behavior, making them react dynamically to player actions.

2. What is procedural content generation (PCG)?
PCG is the use of algorithms to automatically create game content.

3. What challenges does AI introduce in the gaming industry?
AI raises concerns about job displacement and ethical use of generated content.

4. How is AI used in first-person shooters?
AI controls enemy behavior such as movement, attack patterns, and coordination.

5. Why is AI important for immersion?
AI helps create believable and responsive game environments.

## Reflection

The addition of a reranker significantly improved retrival accuracy by prioritizing semantically relevant chunks over purely similar embeddings. Without reranking, unrelated chunks were frequently included. With reranking, the system prioritizes more relevant context, leading to better answers.

The system also correctly handles unknown questions by returning a strict fallback response, ensuring it does not hallucinate information outside the document.
