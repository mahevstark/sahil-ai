import os

import openai


def embed_texts(texts: list, batch_size: int = 100) -> list:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
