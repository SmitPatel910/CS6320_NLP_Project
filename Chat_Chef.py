import numpy as np
import pickle
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from chromadb import EmbeddingFunction, Documents, Embeddings

load_dotenv()
api_key = os.getenv("KEY")
genai_client = genai.Client(api_key=api_key)

with open("saved_embeddings.pkl", "rb") as f:
    formatted_knowledge = pickle.load(f)

embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)
id_map = {}

for i, item in enumerate(formatted_knowledge):
    vector = np.array(item["embedding"], dtype=np.float32)
    faiss_index.add(np.expand_dims(vector, axis=0))
    id_map[i] = {"title": item["title"], "body": item["body"]}


class GeminiEmbed(EmbeddingFunction):
    document_mode = True

    def __call__(self, docs: Documents) -> Embeddings:
        mode = "retrieval_document" if self.document_mode else "retrieval_query"
        response = genai_client.models.embed_content(
            model="models/text-embedding-004",
            contents=docs,
            config=types.EmbedContentConfig(task_type=mode)
        )
        return [e.values for e in response.embeddings]

embed_wrapper = GeminiEmbed()

def generate_recipe_steps(recipe_query: str, top_k: int = 5) -> list:
    try:
        full_query = f"How to make {recipe_query}?"
        embed_wrapper.document_mode = False
        query_vector = embed_wrapper([full_query])[0]
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        distances, indices = faiss_index.search(query_vector, top_k)
        passages = [id_map[i]["title"] for i in indices[0]]

        prompt = f"Based on the below, provide a clear and non-technical response.\n\nQUESTION: {full_query}\n"
        for p in passages:
            prompt += f"REFERENCE: {p.strip()}\n"

        raw_response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        ).text

        raw_response = raw_response.replace("\\n", "\n").replace('\n\n', '\n').strip()
        raw_response = raw_response.replace("*   ", "🔹 ").replace("**", "")
        if "summary" in raw_response.lower():
            raw_response = "📋 " + raw_response
        if "I don't have" in raw_response or "I do not have" in raw_response:
            raw_response = raw_response.replace("I don't have", "⚠️ I don't have").replace("I do not have", "⚠️ I do not have")
        if "let me know" in raw_response.lower():
            raw_response += "\n👉 Let me know what you'd like to hear more about!"

        steps = [s.strip() for s in raw_response.split('\n') if s.strip()]
        return steps

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return []

print(generate_recipe_steps("Brocolli Rice"))