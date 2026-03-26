import os
from dotenv import load_dotenv
import openai
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def main():
    print("Loading document...")

    text = load_document("Selected_Document.txt")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)
    print(f"Document split into {len(chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

    print("Encoding chunks...")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    def search(query):
        q_vec = model.encode([query]).astype("float32")
        _, indices = index.search(q_vec, 20)
        return [(i, chunks[i]) for i in indices[0]]

    def rerank_chunks(question, results, top_m=8):
        pairs = []
        for idx, source in results:
            pairs.append((question, source))

        scores = reranker.predict(pairs)

        scored = []
        for i in range(len(results)):
            idx, source = results[i]
            scored.append((idx, source, scores[i]))

        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                if scored[j][2] > scored[i][2]:
                    temp = scored[i]
                    scored[i] = scored[j]
                    scored[j] = temp

        reranked = []
        limit = min(top_m, len(scored))
        for i in range(limit):
            reranked.append((scored[i][0], scored[i][1]))

        return reranked

    def answer_question(question):
        results = search(question)
        results = rerank_chunks(question, results, 8)
        context = "\n\n".join([source for idx, source in results])

        system_prompt = """
        You are a strict assistant.

        Only answer using the provided context.
        Do NOT add outside knowledge.
        Be specific and concise.
        If the answer is not in the context, respond with EXACTLY: 
        "I don't know. ○|￣|_"
        Do not add punctuiation, emojis, or extra words.
        """

        user_prompt = f"""
        Answer ONLY using the context below.

        Context:
        {context}

        Question: 
        {question}
    
        Answer clearly and directly:
        """

        print("Calling OpenAI...")

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300
        )

        answer = response["choices"][0]["message"]["content"].strip()

        if "i don't know" in answer.lower():
            answer = "I don't know. ○|￣|_"
        return answer, results

    print("\nSystem ready. Ask a question (type 'exit' to quit):")

    while True:
        q = input("Question: ")

        if q.lower() == "exit":
            break

        answer, sources = answer_question(q)

        print("\nAnswer:")
        print(answer)

        if answer.strip() != "I don't know. ○|￣|_":
            print("\n--- SOURCES USED ---")
            for i, (idx, source) in enumerate(sources, 1):
                clean_source = source.strip()
                clean_source = clean_source.replace("\n", " ").strip()

                print(f"\nSource {i} (Chunk {idx}):")

                if len(clean_source) > 250:
                    print(clean_source[:200] + "...")
                else:
                    print(clean_source)

                print("-" * 40)


if __name__ == "__main__":
    main()
