# act_ie/document_store.py

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    import openai
except ImportError:
    openai = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .config import Config
from .models import Document


class DocumentStore:
    """
    In-memory document store with:
    - Dense retrieval via OpenAI embeddings
    - Optional BM25 sparse retrieval for hybrid scoring
    """

    def __init__(self, corpus_csv_path: str = Config.CORPUS_CSV_PATH):
        self.corpus_csv_path = corpus_csv_path
        self.embedding_model = Config.EMBEDDING_MODEL

        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

        # For BM25
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus_tokens: Optional[List[List[str]]] = None

        self._init_openai()
        self._load_corpus()
        self._build_embeddings()
        self._build_bm25()

    def _init_openai(self) -> None:
        if openai is None:
            raise ImportError(
                "openai package is not installed. Install with: pip install openai"
            )
        api_key = os.getenv(Config.OPENAI_API_KEY_ENV)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {Config.OPENAI_API_KEY_ENV} is not set. "
                f"Export your OpenAI key before running."
            )
        openai.api_key = api_key

    def _load_corpus(self) -> None:
        if not os.path.exists(self.corpus_csv_path):
            raise FileNotFoundError(
                f"Corpus CSV not found at {self.corpus_csv_path}. "
                f"Expected columns: doc_id,title,text"
            )

        df = pd.read_csv(self.corpus_csv_path)
        required_cols = {"doc_id", "title", "text"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Corpus CSV must contain columns: {required_cols}. Found: {df.columns}"
            )

        self.documents = [
            Document(str(row["doc_id"]), str(row["title"]), str(row["text"]))
            for _, row in df.iterrows()
        ]
        print(f"[DocumentStore] Loaded {len(self.documents)} documents.")

    def _build_embeddings(self) -> None:
        texts = [doc.text for doc in self.documents]
        if not texts:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            print("[DocumentStore] No documents to embed.")
            return

        resp = openai.Embedding.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [d["embedding"] for d in resp["data"]]
        self.embeddings = np.asarray(vectors, dtype=np.float32)
        print(f"[DocumentStore] Built embeddings with shape {self.embeddings.shape}.")

    def _build_bm25(self) -> None:
        """
        Optional BM25 index for hybrid retrieval.
        """
        if BM25Okapi is None:
            print("[DocumentStore] rank_bm25 not installed; BM25 disabled.")
            return
        corpus_tokens = [doc.text.lower().split() for doc in self.documents]
        self._bm25_corpus_tokens = corpus_tokens
        self._bm25 = BM25Okapi(corpus_tokens)
        print("[DocumentStore] BM25 index built.")

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None or self.embeddings.size == 0:
            return np.zeros(len(self.documents), dtype=np.float32)

        resp = openai.Embedding.create(
            model=self.embedding_model,
            input=[query],
        )
        query_vec = np.asarray(resp["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        return sims

    def _bm25_scores(self, query: str) -> Optional[np.ndarray]:
        if self._bm25 is None or self._bm25_corpus_tokens is None:
            return None
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        return np.asarray(scores, dtype=np.float32)

    def search(self, query: str, top_k: int = Config.TOP_K_DOCS) -> List[Document]:
        """
        Hybrid search: combines dense and BM25 (if available).
        """
        if not self.documents:
            return []

        dense = self._dense_scores(query)
        bm25 = self._bm25_scores(query)

        if bm25 is not None:
            # Simple hybrid: 0.7 * dense + 0.3 * BM25 (normalised)
            dense_norm = (dense - dense.min()) / (dense.ptp() + 1e-8)
            bm25_norm = (bm25 - bm25.min()) / (bm25.ptp() + 1e-8)
            final_scores = 0.7 * dense_norm + 0.3 * bm25_norm
        else:
            final_scores = dense

        top_indices = np.argsort(-final_scores)[:top_k]
        return [self.documents[i] for i in top_indices]
