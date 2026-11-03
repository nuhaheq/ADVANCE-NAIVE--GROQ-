import json
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Safe import for Groq
try:
    from groq import Groq
except ImportError:
    st.error("The 'groq' package is not installed. Make sure requirements.txt is being read by Streamlit Cloud.")
    st.stop()

from langchain_community.vectorstores import FAISS

# Safe import for Hugging Face Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Safe import for CrossEncoder (reranker) — used in "Advanced RAG" mode
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# Safe import for BM25 (lexical retriever) — used in "Hybrid" and "Advanced RAG" modes
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

st.set_page_config(page_title="AFib Clinical Decision Support", layout="wide")
st.title("Retrieval Augmented Generation-Based Clinical Decision Support System for Anticoagulation in Atrial Fibrillation")

TOP_K_PER_STORE = 5        # dense candidates per guideline BEFORE global ranking
DENSE_TOP_N = 20           # dense candidates (global, after dedup) that enter hybrid fusion
BM25_TOP_N = 20            # BM25 candidates (global) that enter hybrid fusion
RRF_K = 60                 # Reciprocal Rank Fusion constant (60 is the value from the original RRF paper)
RERANK_POOL_SIZE = 20      # fused candidates sent to the cross-encoder
TOP_N_CONTEXT = 5          # final number of chunks sent to the LLM

# Q&A log for this session — lost on page refresh/session end, so
# download the CSV BEFORE closing the tab if you want to keep it.
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

# ---------------------------------------------------------------------------
# SIDEBAR: choose the RAG architecture for this query — logged so you can
# compare Naive vs Advanced results once you run the 30 vignettes.
#
#   Naive    : dense (FAISS / L2) top-5
#   Hybrid   : dense + BM25 fused with RRF, top-5            (ablation arm)
#   Advanced : dense + BM25 fused with RRF -> cross-encoder rerank, top-5
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ RAG Architecture Settings")
MODE_NAIVE = "Naive RAG (dense L2 only)"
MODE_HYBRID = "Hybrid RAG (BM25 + dense, RRF)"
MODE_ADVANCED = "Advanced RAG (hybrid + cross-encoder reranking)"
rag_mode = st.sidebar.radio(
    "Select retrieval mode:",
    options=[MODE_NAIVE, MODE_HYBRID, MODE_ADVANCED],
    index=2,
)
use_hybrid = rag_mode in (MODE_HYBRID, MODE_ADVANCED)
use_reranker = rag_mode == MODE_ADVANCED

# Stop instead of silently falling back: a silent fallback would log a run as
# "Advanced" while it actually used a weaker pipeline, corrupting the comparison.
if use_hybrid and BM25Okapi is None:
    st.sidebar.error("The 'rank-bm25' package was not found. Add 'rank-bm25' to requirements.txt.")
    st.stop()
if use_reranker and CrossEncoder is None:
    st.sidebar.error("The 'sentence-transformers' package was not found. Add 'sentence-transformers' to requirements.txt.")
    st.stop()


@st.cache_resource
def setup_and_load_faiss():
    """
    Extract all FAISS ZIP files and load each vectorstore exactly ONCE.

    The same guideline can exist as a top-level ZIP and again inside
    rebuilt_all.zip; stores are de-duplicated by folder name so BM25 statistics
    (IDF) and the dense candidate pool are not inflated by duplicate copies.
    """
    zip_files = sorted(f for f in os.listdir('.') if f.endswith('.zip'))

    for z_file in zip_files:
        folder_name = z_file.replace('.zip', '').split(' ')[0]
        if not os.path.exists(folder_name):
            with zipfile.ZipFile(z_file, 'r') as zip_ref:
                zip_ref.extractall(folder_name)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstores = []
    loaded_names = set()
    for root, dirs, filenames in os.walk('.'):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))  # skip .git etc., deterministic order
        if "index.faiss" in filenames:
            store_name = os.path.basename(os.path.normpath(root))
            if store_name in loaded_names:
                continue
            loaded_names.add(store_name)
            db = FAISS.load_local(root, embedding_model, allow_dangerous_deserialization=True)
            vectorstores.append(db)

    return vectorstores


@st.cache_resource
def load_reranker():
    """Load the cross-encoder reranker once (cached across the session)."""
    if CrossEncoder is None:
        return None
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were will with".split()
)


def tokenize(text: str) -> list:
    """
    Lowercase word tokenizer for BM25. Keeps clinical tokens intact:
    'CHA2DS2-VASc' stays one token, and so do values like '2.5' or '15-29'.
    """
    tokens = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


@st.cache_resource
def build_bm25_index(_vectorstores):
    """
    Build ONE global BM25 index over every chunk in every guideline store.
    (The leading underscore tells Streamlit not to hash the argument.)
    Returns (bm25, docs) where docs[i] is the chunk behind BM25 row i.
    """
    docs = []
    for db in _vectorstores:
        docs.extend(db.docstore._dict.values())
    bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
    return bm25, docs


def bm25_search(query: str, bm25, docs: list, top_n: int) -> list:
    """Top-n chunks by BM25 score (chunks with a zero score are dropped)."""
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [docs[i] for i in top_idx if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list, k: int = RRF_K) -> list:
    """
    Fuse several ranked lists of chunks into one: score(d) = sum over lists of 1 / (k + rank).
    Uses ranks only, so the L2 distances and BM25 scores (different scales) never need
    to be normalised. Chunks are identified by their text (same key as the old dedup).
    """
    fused_scores, first_seen = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.page_content
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, doc)
    ordered = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [first_seen[key] for key in ordered]


def get_guideline_name(metadata: dict) -> str:
    """
    Get the guideline name from a chunk's metadata.
    Tries a few likely key names used by the chunking pipeline
    (the stores in this repo use 'guideline_id').
    """
    for key in ("source_guideline", "source", "guideline", "guideline_id"):
        value = metadata.get(key)
        if value:
            return value
    return "Guideline"


def get_section_name(metadata: dict) -> str:
    for key in ("section", "section_header", "heading"):
        value = metadata.get(key)
        if value:
            return value
    return "General"


def rerank_with_cross_encoder(query: str, docs: list, reranker, top_n: int) -> list:
    """
    Score each (query, chunk) pair with the cross-encoder, then return the
    top_n documents by that score (not the original L2 / RRF order).

    The cross-encoder reads the query AND the chunk together, so it judges
    actual relevance more accurately than embedding distance or rank fusion —
    especially when two chunks look equally close but only one answers the question.
    """
    if reranker is None or not docs:
        return docs[:top_n]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda pair: pair[1], reverse=True)  # cross-encoder score: higher = more relevant

    return [doc for doc, _ in scored_docs[:top_n]]


loaded_vectorstores = setup_and_load_faiss()
reranker_model = load_reranker() if use_reranker else None
bm25_index, bm25_docs = build_bm25_index(loaded_vectorstores) if (use_hybrid and loaded_vectorstores) else (None, [])

if not loaded_vectorstores:
    st.error("No FAISS vectorstore was loaded successfully. Check the ZIP files in this repo.")

case_id = st.text_input(
    "CASE ID (optional, e.g. A01 — used to match Naive vs Advanced rows later):",
    value="",
)
user_query = st.text_area(
    "INSERT CLINICAL QUERY:",
    placeholder="e.g., What is the main treatment used to treat AFib in patients with stage 3 kidney disease?"
)

if st.button("Generate Answer", type="primary"):
    if not user_query.strip():
        st.warning("PLEASE INSERT CLINICAL QUERY FIRST.")
    else:
        with st.spinner("SEARCHING GUIDELINES AND GENERATING ANSWER..."):
            try:
                # 1. Dense retrieval: gather candidates from ALL vectorstores with their distance score.
                #    FAISS's default is L2 distance -> LOWER score = more relevant.
                scored_candidates = []
                for db in loaded_vectorstores:
                    results = db.similarity_search_with_score(user_query, k=TOP_K_PER_STORE)
                    scored_candidates.extend(results)

                # 2. Sort ALL candidates globally by L2 score (most relevant first).
                scored_candidates.sort(key=lambda pair: pair[1])

                # 3. Remove duplicate content, PRESERVE L2 relevance order.
                seen = set()
                dense_ranked = []
                for doc, score in scored_candidates:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        dense_ranked.append(doc)

                # 4. Retrieval mode.
                #    NAIVE    : top-5 by L2 score directly.
                #    HYBRID   : BM25 top-N + dense top-N fused with RRF, top-5 of the fused list.
                #    ADVANCED : same fusion, but a larger fused pool goes to the cross-encoder,
                #               which picks the final top-5.
                bm25_ranked = []
                if use_hybrid:
                    bm25_ranked = bm25_search(user_query, bm25_index, bm25_docs, BM25_TOP_N)
                    fused = reciprocal_rank_fusion([dense_ranked[:DENSE_TOP_N], bm25_ranked])
                    if use_reranker:
                        top_docs = rerank_with_cross_encoder(
                            user_query, fused[:RERANK_POOL_SIZE], reranker_model, TOP_N_CONTEXT
                        )
                    else:
                        top_docs = fused[:TOP_N_CONTEXT]
                else:
                    top_docs = dense_ranked[:TOP_N_CONTEXT]

                context_text = "\n".join([
                    f"- [{get_guideline_name(doc.metadata)} | {get_section_name(doc.metadata)}] {doc.page_content}"
                    for doc in top_docs
                ])

                # RAW chunk list (no guideline/section labels) — this is the list[str]
                # format RAGAS needs for 'retrieved_contexts'. Stored separately from
                # context_text (which has labels, for the LLM/humans to read).
                retrieved_contexts_list = [doc.page_content for doc in top_docs]

                with st.expander("🔍 Debug: retrieved context (check actual metadata key names here)"):
                    st.caption(f"Mode used: **{rag_mode}**")
                    if not top_docs:
                        st.write("No context was found for this query.")
                    dense_rank_of = {d.page_content: i for i, d in enumerate(dense_ranked, start=1)}
                    bm25_rank_of = {d.page_content: i for i, d in enumerate(bm25_ranked, start=1)}
                    for i, doc in enumerate(top_docs, start=1):
                        st.markdown(f"**Chunk {i}** — metadata keys: `{list(doc.metadata.keys())}`")
                        if use_hybrid:
                            st.caption(
                                f"dense rank: {dense_rank_of.get(doc.page_content, '-')} | "
                                f"BM25 rank: {bm25_rank_of.get(doc.page_content, '-')}"
                            )
                        st.json(doc.metadata)
                        preview = doc.page_content[:300]
                        st.caption(preview + ("..." if len(doc.page_content) > 300 else ""))

                system_prompt = f"""
You are a Board-Certified Clinical Specialist Professor in Atrial Fibrillation (AFib) Pharmacotherapy.

YOUR OBJECTIVE:
Provide structured, highly accurate clinical recommendations in response to the user's query using ONLY the provided official guideline context chunks. You must attribute every clinical statement to its exact source guideline and section header.

==================================================
STRICT CLINICAL SAFETY RULES:
==================================================
1. STRICT GROUNDING: Base every recommendation solely on the CONTEXT DATA below. Do NOT use outside medical knowledge or unvalidated general assumptions.
2. ABSENCE OF EVIDENCE: If the provided CONTEXT DATA does not contain sufficient information to safely answer the user's query, explicitly state: "The retrieved guideline context does not contain sufficient clinical guidance to answer this specific query."
3. MANDATORY IN-LINE CITATIONS: Every clinical recommendation, dose, risk-score threshold, or contraindication MUST be followed immediately by an inline bracketed citation containing the Guideline ID and Section Name.
   - Format: [Guideline_ID | Section_Header]
   - Example: "Apixaban 5 mg twice daily is recommended for stroke prevention [ACC_AHA_2023 | 4.1 Stroke Prevention]."
4. NO UNVALIDATED RATIONALES: Only cite active, validated recommendations.
5. CONTRADICTION HANDLING: If multiple guidelines give conflicting recommendations, explicitly compare both views clearly.

==================================================
CONTEXT DATA FROM GUIDELINES:
==================================================
{context_text}

==================================================
RESPONSE STRUCTURE:
==================================================
Structure your response using the following format:

**Clinical Recommendation Summary**
- Provide a direct 2-3 sentence summary answering the query with inline citations.

**Detailed Guideline Guidance**
- Break down specific dosing, risk stratification scores, renal adjustments, or therapeutic choices using bullet points.
- Ensure EVERY bullet point has its corresponding [Guideline_ID | Section_Header] citation.

**Guideline Provenance Matrix**
Provide a Markdown table summarizing all guidelines cited:

| Guideline / Source | Section Referenced | Key Recommendation Summary |
| :--- | :--- | :--- |

==================================================
USER QUERY: {user_query}

ANSWER:
"""

                # CHANGE THIS FOR THE GROQ MODEL
                api_key = st.secrets["GROQ_API_KEY"]
                client = Groq(api_key=api_key)

                # List of currently supported official Groq models
                candidate_models = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "openai/gpt-oss-120b",
                    "gpt-oss"
                ]

                answer_text = None
                model_used = None
                last_error = None

                for model_id in candidate_models:
                    try:
                        completion = client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": system_prompt}],
                        )
                        answer_text = completion.choices[0].message.content
                        model_used = model_id
                        st.caption(f"🤖 Answer generated using model: `{model_id}` | RAG mode: `{rag_mode}`")
                        break  # Success! Exit the loop
                    except Exception as e:
                        last_error = e
                        continue

                if answer_text is None:
                    st.error(f"An error occurred: {str(last_error)}")
                else:
                    st.markdown(answer_text)

                    # Log to the session — 'rag_mode' is important for filtering/comparing
                    # Naive vs Advanced results after you download the CSV.
                    #
                    # 'retrieved_contexts' is stored as a JSON string (list[str]) — this is
                    # the exact format the RAGAS library needs to compute Faithfulness,
                    # Context Precision, and Context Recall. Without this column, those
                    # metrics CANNOT be computed even if you have an answer and a ground
                    # truth. 'ground_truth' is the 'reference' RAGAS needs for Context
                    # Recall — fill it in after downloading.
                    #
                    # 'model_id' records which Groq model actually answered (the loop above can
                    # fall back to a different model), so you can check that Naive and Advanced
                    # rows were generated by the same model.
                    st.session_state.qa_log.append({
                        "case_id": case_id.strip(),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "rag_mode": rag_mode,
                        "model_id": model_used,
                        "query": user_query,
                        "answer": answer_text,
                        "retrieved_contexts": json.dumps(retrieved_contexts_list, ensure_ascii=False),
                        "retrieved_contexts_count": len(retrieved_contexts_list),
                        "ground_truth": "",
                    })

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# This Session's Q&A Log
if st.session_state.qa_log:
    st.divider()
    st.subheader(f"📊 This Session's Q&A Log ({len(st.session_state.qa_log)} records)")
    df_log = pd.DataFrame(st.session_state.qa_log)
    st.dataframe(df_log, use_container_width=True)

    csv_bytes = df_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Log as CSV (eval_vignettes.csv)",
        data=csv_bytes,
        file_name="eval_vignettes.csv",
        mime="text/csv",
    )
    st.caption(
        "The `ground_truth` column is empty (and `case_id` too, if you left it blank) — fill them in "
        "in Excel after downloading (match `case_id` to your vignette table, e.g. A01, B03). "
        "`rag_mode` shows which architecture was used for each row — important for your "
        "Naive vs Advanced RAG comparison. `model_id` shows which Groq model generated the answer. "
        "`retrieved_contexts` is the list of chunks the system actually retrieved (JSON format) — "
        "do not delete or edit this column; RAGAS reads it directly to compute Faithfulness, "
        "Context Precision, and Context Recall. To use it with the ragas library: `json.loads()` "
        "this column first to convert it back into a Python list before passing it to RAGAS."
    )
