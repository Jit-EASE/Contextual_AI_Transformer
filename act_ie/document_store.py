# act_ie/document_store.py

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from openai import OpenAI
from .config import Config
from .models import Document


class DocumentStore:
    """
    Document store with OpenAI v1+ embeddings + optional BM25 hybrid search.
    """

    def __init__(self, corpus_csv_path: str = Config.CORPUS_CSV_PATH):
        self.client = OpenAI()   # NEW v1 client
        self.corpus_csv_path = corpus_csv_path
        self.embedding_model = Config.EMBEDDING_MODEL

        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load_corpus()
        self._build_embeddings()
        self._build_bm25()

    def _load_corpus(self) -> None:
        if not os.path.exists(self.corpus_csv_path):
            raise FileNotFoundError(
                f"Corpus CSV not found at {self.corpus_csv_path}"
            )

        df = pd.read_csv(self.corpus_csv_path)
        required_cols = {"doc_id", "title", "text"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Corpus CSV missing required columns: {required_cols}")

        self.documents = [
            Document(str(row["doc_id"]), str(row["title"]), str(row["text"]))
            for _, row in df.iterrows()
        ]

    def _build_embeddings(self) -> None:
        texts = [doc.text for doc in self.documents]
        if not texts:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            return

        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        vectors = [item.embedding for item in resp.data]
        self.embeddings = np.asarray(vectors, dtype=np.float32)

    def _build_bm25(self) -> None:
        if BM25Okapi is None:
            return
        tokens = [doc.text.lower().split() for doc in self.documents]
        self._bm25 = BM25Okapi(tokens)
        self._bm25_tokens = tokens

    def _dense_scores(self, query: str):
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        q_vec = np.asarray(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        return cosine_similarity(q_vec, self.embeddings)[0]

    def _bm25_scores(self, query: str):
        if BM25Okapi is None:
            return None
        return np.asarray(self._bm25.get_scores(query.lower().split()), dtype=np.float32)

    def search(self, query: str, top_k: int = Config.TOP_K_DOCS):
        dense = self._dense_scores(query)
        bm25 = self._bm25_scores(query)

        if bm25 is not None:
            d = (dense - dense.min()) / (dense.ptp() + 1e-8)
            b = (bm25 - bm25.min()) / (bm25.ptp() + 1e-8)
            scores = 0.7 * d + 0.3 * b
        else:
            scores = dense

        idx = np.argsort(-scores)[:top_k]
        return [self.documents[i] for i in idx]
