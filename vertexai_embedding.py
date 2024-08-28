import json
import logging
import sys
import numpy as np
import faiss
from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

GEMINI_EMBEDDING_MODEL = "text-multilingual-embedding-002"
JSON_FILE_NAME = "./KO_08_12_2024.json"
FAISS_DATABASE_NAME = "./KO_08_12_2024.faiss"

DIMENSION = 768
BATCH_SIZE = 25

def load_chunks_from_json() -> List[str] :
    try:
        with open(JSON_FILE_NAME, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = [dic['content'] for dic in data]
        logger.info(f"Loaded {len(chunks)} contents from JSON file")
        return chunks

    except FileNotFoundError:
        logger.error(f"File {JSON_FILE_NAME} not found")
        sys.exit(1)

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file {JSON_FILE_NAME}")
        sys.exit(1)

def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in texts]
        kwargs = dict(output_dimensionality=DIMENSION)
        embeddings = model.get_embeddings(inputs, **kwargs)
        return [embedding.values for embedding in embeddings]
    except Exception as e:
        logger.error(f"Error occurred while generating embeddings: {e}")
        sys.exit(1)

def commit_faiss(index):
    try:
        faiss.write_index(index, FAISS_DATABASE_NAME)
        logger.info(f"FAISS index saved. Total vectors: {index.ntotal}")
        print(f"Processing completed. Total vectors: {index.ntotal}")

    except Exception as e:
        logger.error(f"Error occurred while saving FAISS index: {e}")
        sys.exit(1)

def main():
    chunks = load_chunks_from_json()
    index = faiss.IndexFlatL2(DIMENSION)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        try:
            result = embed_texts(texts=batch)
            embedding_array = np.array(result, dtype=np.float32)
            index.add(embedding_array)
            logger.info(f"Processed batch {i//BATCH_SIZE + 1}. Total vectors: {index.ntotal}")

        except Exception as e:
            logger.error(f"Error occurred while generating embeddings: {e}")
            sys.exit(1)

    commit_faiss(index)

if __name__ == "__main__":
    main()