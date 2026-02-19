# MVP Requirements Specification

## Public RAG Pipeline on Epstein Files

**Purpose**
Build a **no‑cost, publicly accessible MVP** of a Retrieval‑Augmented Generation (RAG) system over the Epstein Files that demonstrates strong AI product thinking, transparency, and practical usefulness for a general LinkedIn audience.

The MVP should be:

* Free to access (no login, no paywall)
* Educational and credible (grounded answers, citations)
* Technically sound but lightweight
* Reusable as a reference architecture for other document corpora

---

## 1. Target Audience

Primary:

* LinkedIn audience (product managers, AI practitioners, researchers, tech leaders)

Secondary:

* Journalists / researchers exploring public records
* Students learning RAG architectures

Non‑Goals:

* This is **not** a legal analysis tool
* This is **not** investigative journalism
* This is **not** a production‑grade enterprise system

---

## 2. Core User Value Proposition

Users should be able to:

1. Ask natural‑language questions about the Epstein Files
2. Get **fact‑grounded answers only** from retrieved documents
3. See **exact source snippets** that support each answer
4. Explore large, messy public records **without reading thousands of pages**

---

## 3. Functional Requirements

### 3.1 Data Ingestion

**Scope (MVP):**

* **Source:** [teyler/epstein-files-20k](https://huggingface.co/datasets/teyler/epstein-files-20k) on Hugging Face
* Pre‑curated text data from the Epstein Files

**Requirements:**

* One‑time offline ingestion from Hugging Face
* Support for:

  * Parsing the dataset structure (likely Parquet/Arrow via HF datasets library)
  * Metadata mapping from dataset columns
* Extract and store metadata:

  * Document source
  * Date (if available)
  * File / section identifier

**Out of Scope (MVP):**

* OCR
* PDFs / images
* Continuous live updates

---

### 3.2 Chunking & Pre‑Processing

**Requirements:**

* Semantic chunking (not fixed‑size only)
* Chunk size configurable (default ~500–800 tokens)
* Remove:

  * Empty or boilerplate text
  * Duplicate chunks
* Each chunk must retain metadata links to original document

---

### 3.3 Embeddings & Vector Store

**Requirements:**

* Use open‑source embedding model (cost‑free)
* Vector store must:

  * Support semantic similarity search
  * Run locally or via free tier

**Recommended:**

* Chroma / FAISS (single‑node)

**Out of Scope:**

* Paid vector DBs
* Multi‑tenant scaling

---

### 3.4 Retrieval Strategy

**MVP Retrieval Flow:**

1. User query
2. Dense vector search (top‑k)
3. Optional diversity (MMR or simple deduplication)

**Requirements:**

* Top‑k configurable (default 5–8)
* Retrieval must return:

  * Chunk text
  * Source metadata

---

### 3.5 Answer Generation (LLM)

**Requirements:**

* Use a free or free‑tier LLM
* Prompt must:

  * Answer **only from retrieved context**
  * Say "Not found in documents" if answer is missing
  * Avoid speculation or summarization beyond evidence

**Output Format:**

* Direct answer
* Bullet list of cited sources

---

### 3.6 User Interface (Web)

**MVP UI Requirements:**

* Public web app (no login)
* Single‑page experience
* Components:

  * Question input box
  * Answer panel
  * Expandable "Sources" section

**Nice‑to‑Have:**

* Example questions users can click
* Loading indicator during retrieval

**Explicitly Excluded:**

* User accounts
* History saving
* Personalization

---

## 4. Non‑Functional Requirements

### 4.1 Cost Constraints

* Zero infrastructure cost preferred
* Use:

  * Local execution OR
  * Free tiers only

### 4.2 Performance

* Response time target: < 10 seconds
* Dataset size: up to a few million lines (static)

### 4.3 Reliability

* Graceful failure if model or retrieval fails
* Clear user messaging when answer cannot be generated

---

## 5. Safety, Ethics & Disclaimers

**Mandatory:**

* Disclaimer banner:

  > "This tool answers questions based only on publicly available documents. It does not make claims, judgments, or allegations."

**Model Guardrails:**

* No speculation
* No defamation
* No synthesis beyond source text

---

## 6. Transparency & Trust Features

**Required:**

* Every answer must show:

  * Source document name
  * Exact supporting text snippet

**Nice‑to‑Have:**

* Highlight retrieved text used for answer

---

## 7. Evaluation (Lightweight)

**MVP Checks:**

* Manual test set of ~20 questions
* Validate:

  * Correct retrieval
  * Faithful answers
  * No hallucinations

---

## 8. Deliverables from Antigravity

Antigravity should deliver:

1. End‑to‑end runnable RAG pipeline
2. Publicly accessible web UI
3. Clear README covering:

   * Architecture overview
   * How data was processed
   * Known limitations
4. Simple configuration file for:

   * Model selection
   * Chunk size
   * Top‑k retrieval

---

## 9. Success Criteria (MVP)

The MVP is successful if:

* A LinkedIn user can open the link and ask a question with no setup
* Answers are grounded and cite sources
* The system does not hallucinate or speculate
* The project clearly demonstrates **applied RAG product thinking**

---

## 10. Future Enhancements (Post‑MVP)

(Not required now, but should be architecturally possible)

* Hybrid search (BM25 + vectors)
* PDF/OCR ingestion
* Multi‑dataset support
* Retrieval quality metrics dashboard
* Multi‑hop question answering

---

**End of Requirements Specification**
