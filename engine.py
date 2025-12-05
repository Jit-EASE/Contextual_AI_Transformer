# act_ie/engine.py

from typing import List

from .config import Config
from .document_store import DocumentStore
from .llm_backend import LLMBackend
from .models import Document, NumericContext, EngineAnswer, AnswerMeta
from .numeric_context import summarise_numeric_features
from .geo_policy import lookup_geo_policy
from .compliance import classify_risk_level, build_audit_entry, audit_log


class ActIEEngine:
    """
    ACT-IE v0/v1 core engine: Retrieval + numeric context + geo-policy + compliance → LLM.
    """

    def __init__(
        self,
        corpus_csv_path: str = Config.CORPUS_CSV_PATH,
        llm_backend: str = Config.LLM_BACKEND,
    ):
        self.doc_store = DocumentStore(corpus_csv_path=corpus_csv_path)
        self.llm = LLMBackend(backend=llm_backend)

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

You must:
- Use the numeric and regional context.
- Use the retrieved documents as primary evidence.
- Consider geo-policy metadata where available (nitrates bands, ACRES zones, etc.).
- Reflect the structure of Irish and EU policy where relevant, including (non-exhaustive):
  * Food Vision 2030 (Ireland)
  * CAP Strategic Plan for Ireland 2023–2027 (Pillar I and II)
  * EU AI Act, Data Act, Data Governance Act, GDPR, and relevant environmental directives
- Be explicit about uncertainty and data gaps.
- Avoid overconfident forecasts when the evidence is weak.
- Clearly separate FACTS from INTERPRETATION and SCENARIO.

------------------------------
NUMERIC & REGIONAL CONTEXT
------------------------------
{numeric_summary}

------------------------------
GEO-POLICY CONTEXT (STUB)
------------------------------
Nitrates band: {geo_policy.get('nitrates_band')}
ACRES zone: {geo_policy.get('acres_zone')}
Notes: {geo_policy.get('notes')}

------------------------------
RETRIEVED POLICY & RESEARCH CONTEXT
------------------------------
{docs_block}

------------------------------
USER QUESTION
------------------------------
{query}

------------------------------
RESPONSE STYLE
------------------------------
1. Start with a 2–3 sentence executive summary in clear, formal English.
2. Then provide:
   - A. Data- and policy-grounded analysis for {sector} in {county}.
   - B. Risks, uncertainties, and missing data.
   - C. Strategic options or scenarios (short-term and medium-term).
3. Where appropriate, link implications to farmers, co-ops, processors, and policymakers.
4. End with a one-line disclaimer noting that this is a model-based analytical perspective,
   not legal or financial advice.

Now produce the answer.
"""
        return prompt.strip()

    def answer(
        self,
        query: str,
        numeric: NumericContext,
        county: str,
        sector: str,
        top_k_docs: int = Config.TOP_K_DOCS,
        max_tokens: int = 450,
    ) -> EngineAnswer:
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

        # Audit log
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
