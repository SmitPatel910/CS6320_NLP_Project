import numpy as np
import pickle
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv
import os
from chromadb import EmbeddingFunction, Documents, Embeddings
import re

load_dotenv(find_dotenv())
api_key = os.getenv("KEY")
genai_client = genai.Client(api_key=api_key)

embedding_path = 'rag_gemini_instructionGenerator/saved_embeddings.pkl'
with open(embedding_path, "rb") as f:
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
        # Step 1: Prepare the query embedding
        full_query = f"How to make {recipe_query}?"
        embed_wrapper.document_mode = False
        query_vector = embed_wrapper([full_query])[0]
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Step 2: Search in FAISS index
        distances, indices = faiss_index.search(query_vector, top_k)
        references = [id_map[i]["title"] for i in indices[0]]

        # Step 3: Build the prompt
        prompt = f"""You are a recipe expert.

Only return a list of **numbered cooking steps** (e.g., 1. Do this, 2. Do that) to make: {recipe_query}.

Do not include titles, references, summaries, general guides, introductions, or explanations — only the cooking steps.

Use the following references as inspiration:

"""
        for ref in references:
            prompt += f"{ref.strip()}\n"

        # Step 4: Query the Gemini model
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        ).text.strip()

        # Step 5: Clean response: remove markdown bold and extra numbering
        lines = response.splitlines()
        steps = []

        for line in lines:
            line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)  # remove markdown bold
            line = re.sub(r"^\d+\.\s*", "", line)          # remove leading number
            line = re.sub(r"^[-•*]\s*", "", line)          # remove bullets
            if line.strip():
                steps.append(line.strip())

        # Step 6: Add correct numbering
        steps = [f"{i+1}. {step}" for i, step in enumerate(steps)]

        return steps

    except Exception as e:
        print(f"Error generating recipe steps: {e}")
        return []
