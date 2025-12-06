# act_ie/engine.py

from typing import List, Optional

from .config import Config
from .document_store import DocumentStore
from .llm_backend import LLMBackend
from .models import Document, NumericContext, EngineAnswer, AnswerMeta
from .numeric_context import summarise_numeric_features
from .geo_policy import lookup_geo_policy
from .compliance import classify_risk_level, build_audit_entry, audit_log


class ActIEEngine:
    """
    Production-safe ACT-IE engine.
    Heavy components are lazy-loaded so server startup doesn't fail.
    """

    def __init__(
        self,
        corpus_csv_path: str = Config.CORPUS_CSV_PATH,
        llm_backend: str = Config.LLM_BACKEND,
    ):
        self.corpus_csv_path = corpus_csv_path
        self.llm_backend = llm_backend

        self.doc_store: Optional[DocumentStore] = None
        self.llm: Optional[LLMBackend] = None
        self.embeddings_ready: bool = False

    # ----------------------------------------------------
    # Lazy initialisation
    # ----------------------------------------------------
    def _ensure_initialized(self):
        if self.doc_store is None:
            self.doc_store = DocumentStore(
                corpus_csv_path=self.corpus_csv_path,
                auto_init_openai=False,       # <-- IMPORTANT
                auto_build_embeddings=False,  # <-- IMPORTANT
            )

        if self.llm is None:
            self.llm = LLMBackend(backend=self.llm_backend)

    # ----------------------------------------------------
    # Manual embedding build (triggered by API route)
    # ----------------------------------------------------
    def build_embeddings(self):
        self._ensure_initialized()
        self.doc_store.init_openai()
        self.doc_store.build_embeddings()
        self.embeddings_ready = True

    # ----------------------------------------------------
    # Prompt construction
    # ----------------------------------------------------
    def _build_prompt(
        self,
        query: str,
        numeric_summary: str,
        retrieved_docs: List[Document],
        county: str,
        sector: str,
    ) -> str:
        docs_str_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            snippet = doc.text
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "... [truncated]"
            docs_str_parts.append(
                f"[DOC {i}] {doc.title} (ID={doc.doc_id})\n{snippet}"
            )
        docs_block = "\n\n".join(docs_str_parts) if docs_str_parts else "(No documents retrieved)."

        geo_policy = lookup_geo_policy(county, sector)

        prompt = f"""
You are ACT-IE, a contextual AI specialised in the Irish agri-food sector.
[… unchanged …]
"""
        return prompt.strip()

    # ----------------------------------------------------
    # Main answer method
    # ----------------------------------------------------
    def answer(
        self,
        query: str,
        numeric: NumericContext,
        county: str,
        sector: str,
        top_k_docs: int = Config.TOP_K_DOCS,
        max_tokens: int = 450,
    ) -> EngineAnswer:

        self._ensure_initialized()

        if not self.embeddings_ready:
            raise RuntimeError(
                "Embeddings not built. Call POST /initialize first."
            )

        numeric_summary = summarise_numeric_features(numeric, county, sector)

        retrieval_query = f"{query} | {county} | {sector} | Irish agri-food policy"
        retrieved_docs = self.doc_store.search(retrieval_query, top_k=top_k_docs)

        risk_assessment = classify_risk_level(query)

        prompt = self._build_prompt(
            query=query,
            numeric_summary=numeric_summary,
            retrieved_docs=retrieved_docs,
            county=county,
            sector=sector,
        )

        answer_text = self.llm.generate(prompt, max_tokens=max_tokens)

        meta = AnswerMeta(
            county=county,
            sector=sector,
            retrieved_doc_ids=[d.doc_id for d in retrieved_docs],
            risk_assessment=risk_assessment,
            numeric_summary=numeric_summary,
        )

        log_entry = build_audit_entry(
            query=query,
            county=county,
            sector=sector,
            numeric_summary=numeric_summary,
            retrieved_doc_ids=meta.retrieved_doc_ids,
            risk_assessment=risk_assessment,
            answer=answer_text,
        )
        audit_log(log_entry)

        return EngineAnswer(answer=answer_text, meta=meta)
