<div align="center">

# JobNexusRAG

### Job Recommendation Engine built on NexusRAG (Hybrid RAG + Knowledge Graph)  
**Author:** Dang Huynh Son — forked from NexusRAG by Le Duc Dat

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Upload CVs and Job Descriptions. Get ranked, explainable matches.**

JobNexusRAG is a **two‑sided job recommendation engine** (candidates ↔ jobs) built on top of the original **NexusRAG** pipeline:

- Deep parsing of CVs / JDs with **Docling**
- Hybrid retrieval with **vector search + knowledge graph**
- **Cross‑encoder reranking** and multi‑factor scoring
- Streaming **LLM chat assistant** for jobs and skills

[What is JobNexus?](#overview) · [Architecture](#architecture) · [Quick Start](#quick-start) · [API](#api) · [Tech Stack](#tech-stack)

</div>

---

## Overview

### What JobNexus does

JobNexus turns the generic document‑Q&A system NexusRAG into a **job platform engine**:

- **Candidate side**
  - Upload CVs (PDF/DOCX/TXT/MD)
  - LLM extracts skills, experience, education, target role
  - Get **ranked job recommendations** with score breakdowns and skill gap insights

- **Recruiter side**
  - Create **companies** and **job postings** (structured fields + optional JD file)
  - LLM extracts required skills, experience level, location, salary range
  - Get **ranked candidate recommendations** for each job

- **Knowledge graph**
  - Built with **LightRAG** over CV + JD markdown
  - Entities: Skill, JobTitle, Company, Industry, Certification, University, Location, Technology, Person
  - Used to enrich skill matching and provide explainable recommendations

### High‑level architecture

- **Ingestion**
  - CV / JD file → Docling deep parsing → HybridChunker
  - LLM structured extraction (skills, experience, education, requirements)
  - Chunk embeddings stored in **ChromaDB** collections:
    - `cv_chunks` for candidate chunks
    - `jd_chunks` for job chunks
  - Markdown content ingested into **LightRAG** knowledge graph

- **Matching**
  - Candidate → Jobs:
    - Query `jd_chunks` using CV summary + skills embedding
  - Job → Candidates:
    - Query `cv_chunks` using JD requirements embedding
  - Cross‑encoder reranking (BAAI/bge‑reranker‑v2‑m3)
  - Multi‑factor scoring:
    - Semantic similarity
    - Skill match
    - Experience fit
    - Location fit
    - Salary overlap

- **Frontend**
  - **Candidate Dashboard**: manage profile, upload/process CV, view job matches
  - **Recruiter Dashboard**: manage companies/jobs, view candidate matches
  - **Chat**: ask questions about jobs, skills, and career paths (reuses NexusRAG chat)

---

## Architecture

### End‑to‑end flow

1. **CV ingestion**
   - Candidate uploads a CV file
   - Docling parses and converts to enriched markdown
   - Gemini (or your configured LLM) extracts:
     - Skills (technical + soft)
     - Work experience (company, role, duration, description)
     - Education (school, degree, field, year)
     - Target role, years of experience, location
   - Parsed chunks are embedded with **BAAI/bge‑m3** into `cv_chunks` collection in ChromaDB
   - Markdown is ingested into **LightRAG** to populate the skills knowledge graph

2. **Job ingestion**
   - Recruiter creates a job posting (structured form) and/or uploads a JD file
   - Docling parses and converts to markdown (for files)
   - LLM extracts:
     - Required skills vs nice‑to‑have skills
     - Experience level
     - Location
     - Salary range (if present)
   - Parsed chunks are embedded into `jd_chunks` collection in ChromaDB
   - Markdown is ingested into the same LightRAG knowledge graph

3. **Candidate → Jobs matching**
   - Build a query from:
     - Extracted CV summary
     - Top skills
     - Desired role
   - Embed query and **over‑fetch** top‑K job chunks from `jd_chunks`
   - Cross‑encoder rerank candidate chunks vs query
   - Compute structured scores:
     - Skill match (Jaccard over required skills, optionally expanded via KG)
     - Experience fit (years vs required, smoothed via sigmoid)
     - Location fit (same city, remote, or mismatch)
     - Salary fit (range overlap)
   - Store and return **MatchResult** entities with explanations and score breakdown

4. **Job → Candidates matching**
   - Build a query from:
     - Job title
     - Required skills
     - JD summary
   - Embed query and **over‑fetch** top‑K candidate chunks from `cv_chunks`
   - Cross‑encoder rerank candidate chunks vs query
   - Compute the same structured scores and persist **MatchResult**

5. **Knowledge graph**
   - LightRAG runs asynchronously over the combined CV + JD markdown
   - Exposes:
     - `get_entities` / `get_relationships` / `get_graph_data`
     - `get_relevant_context(question)` for KG‑only context injection
   - Used to:
     - Expand skills (e.g. Docker ↔ Kubernetes)
     - Provide explainable “related skills” in match explanations

---

## Features

### Candidate experience

- **Candidate profiles**
  - Name, email, phone, location
  - Desired role, desired salary range
  - Total years of experience, education level

- **CV management**
  - Upload multiple CV versions per candidate
  - Status tracking: `pending → parsing → indexing → indexed / failed`
  - Extracted skills and summary visible in the UI

- **Job recommendations**
  - Ranked list of jobs with:
    - Overall match score (%)
    - Breakdown: semantic / skills / experience / location / salary
    - Matched and missing skills badges
  - Click to expand for a detailed score bar visualization

### Recruiter experience

- **Company & job management**
  - Company profiles (name, industry, location, size, description)
  - Job postings:
    - Title, location
    - Salary min/max
    - Required and nice‑to‑have skills
    - Experience level
    - Optional JD file upload

- **Candidate recommendations**
  - Ranked list of candidates for each job
  - Same score breakdown and skill badges
  - Quick navigation between jobs and their candidate lists

### Chat assistant (optional)

JobNexus keeps the original NexusRAG **agentic chat**:

- SSE streaming with agent steps (“Analyzing → Retrieving → Generating → Done”)
- Function calling backed by the hybrid retriever
- Used for:
  - Asking about required skills for a role
  - Exploring skill gaps
  - Explaining why a job or candidate is a good fit

---

## Data model

### Core entities (backend models)

- `Candidate`
  - Basic profile details + preferences
  - One‑to‑many with `CandidateCV`

- `CandidateCV`
  - File metadata (name, size, type)
  - Parsing/indexing status
  - Extracted skills, experience, education, summary
  - Chunk and page counts

- `Company`
  - Name, industry, location, size, description
  - One‑to‑many with `JobPosting`

- `JobPosting`
  - Company reference
  - Title, description text, location
  - Salary range
  - Required and nice‑to‑have skills
  - Experience requirement
  - Optional JD file metadata
  - Chunk count and processing time

- `MatchResult`
  - Candidate ↔ Job pair
  - Overall score
  - `semantic_score`, `skill_match_score`, `experience_score`,
    `location_score`, `salary_score`
  - Matched / missing skills
  - Free‑text explanation

---

## Quick Start

JobNexusRAG reuses the original NexusRAG tooling and Docker setup. You can run it either with Docker (recommended) or in local dev mode.

### Option A: Docker (Full stack)

```bash
git clone https://github.com/huynhsown/JobNexusRAG.git
cd JobNexusRAG
copy .env.example .env        # Windows
# or: cp .env.example .env    # Linux / macOS

# Edit .env — set GOOGLE_AI_API_KEY (or switch to Ollama)
docker compose up --build
```

First build can take several minutes (downloads ML models ~2.5 GB).

- Backend: `http://localhost:8080`
- Frontend: `http://localhost:5174`

### Option B: Local development

You can also run backend and frontend separately without Docker.

#### 1. Backend (FastAPI)

```bash
cd backend
py -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
``+

Make sure:

- PostgreSQL is running on `localhost:5433` (or adjust `DATABASE_URL` in `.env`)
- ChromaDB is running on `localhost:8002` (or adjust `CHROMA_HOST` / `CHROMA_PORT`)

Run the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

#### 2. Frontend (React/Vite)

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5174
```

Open `http://localhost:5174` in your browser.

---

## Using JobNexus

### Candidate flow (Find Jobs)

1. Open the app: `http://localhost:5174`
2. Click **“Find Jobs”** on the home page
3. In the **Candidate Dashboard**:
   - Create a new candidate profile (name, basic info)
   - Upload a CV file (PDF/DOCX/TXT/MD)
   - Click **Process** to parse + extract + index the CV
   - After status is `indexed`, click **“Find Matching Jobs”**
4. Inspect results:
   - Overall match score
   - Score breakdown (semantic / skills / experience / location / salary)
   - Matched / missing skills badges

### Recruiter flow (Find Talent)

1. Click **“Find Talent”** on the home page
2. In the **Recruiter Dashboard**:
   - Create a company (if not already created)
   - Create a job posting with title, location, skills and optional description
   - (Optional) Upload a JD file and click **Process** to parse + index
   - Click **“Find Matching Candidates”**
3. Inspect candidate matches similarly to the candidate view.

---

## Configuration

Copy `.env.example` and adjust values:

```bash
cp .env.example .env    # or use `copy` on Windows
```

### Required

| Variable | Description |
|---|---|
| `GOOGLE_AI_API_KEY` | Google AI API key (required if `LLM_PROVIDER=gemini`) |

### LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `ollama` |
| `LLM_MODEL_FAST` | `gemini-2.5-flash` | Model for chat and KG extraction |
| `LLM_THINKING_LEVEL` | `medium` | Gemini 3.x thinking: `minimal` / `low` / `medium` / `high` |
| `LLM_MAX_OUTPUT_TOKENS` | `8192` | Max output tokens (includes thinking) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:12b` | Ollama model name |

### KG embedding

| Variable | Default | Description |
|---|---|---|
| `KG_EMBEDDING_PROVIDER` | `gemini` | `gemini`, `ollama`, or `sentence_transformers` |
| `KG_EMBEDDING_MODEL` | `gemini-embedding-001` | Model name (provider‑specific) |
| `KG_EMBEDDING_DIMENSION` | `3072` | Embedding dimension (must match model) |

### RAG + matching pipeline

| Variable | Default | Description |
|---|---|---|
| `NEXUSRAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Text embedding model (1024‑dim) |
| `NEXUSRAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross‑encoder reranker |
| `NEXUSRAG_VECTOR_PREFETCH` | `20` | Candidates over‑fetched before reranking |
| `NEXUSRAG_RERANKER_TOP_K` | `8` | Final results after reranking |
| `NEXUSRAG_ENABLE_KG` | `true` | Enable knowledge graph extraction |
| `NEXUSRAG_ENABLE_IMAGE_EXTRACTION` | `true` | Extract images from documents |
| `NEXUSRAG_ENABLE_IMAGE_CAPTIONING` | `true` | Caption images via LLM for search |
| `NEXUSRAG_KG_LANGUAGE` | `Vietnamese` | KG extraction language (can be `English` or multilingual) |

### Matching weights

JobNexus exposes weights for each score component:

| Variable | Default | Description |
|---|---|---|
| `MATCHING_SEMANTIC_WEIGHT` | `0.50` | Weight for semantic similarity |
| `MATCHING_SKILL_WEIGHT` | `0.25` | Weight for skill match |
| `MATCHING_EXPERIENCE_WEIGHT` | `0.10` | Weight for experience fit |
| `MATCHING_LOCATION_WEIGHT` | `0.10` | Weight for location fit |
| `MATCHING_SALARY_WEIGHT` | `0.05` | Weight for salary fit |

You can tune these in `.env` to emphasize different aspects.

---

## API

All endpoints are prefixed with `/api/v1`. Interactive docs: `http://localhost:8080/docs`.

### Candidates

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/candidates` | List all candidates |
| `POST` | `/candidates` | Create a candidate profile |
| `GET` | `/candidates/{id}` | Get candidate with CVs and extracted data |
| `PUT` | `/candidates/{id}` | Update candidate profile |
| `DELETE` | `/candidates/{id}` | Delete candidate + CVs + chunks |
| `POST` | `/candidates/{id}/upload-cv` | Upload a CV file |
| `POST` | `/candidates/{id}/process/{cv_id}` | Trigger CV processing |
| `GET` | `/candidates/{id}/recommendations` | Get job recommendations for a candidate |

### Jobs & companies

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/jobs/companies` | Create a company |
| `GET` | `/jobs/companies` | List companies |
| `GET` | `/jobs/companies/{company_id}` | Get company details |
| `POST` | `/jobs` | Create a job posting |
| `GET` | `/jobs` | List jobs (optional filters) |
| `GET` | `/jobs/{id}` | Get job details (with company) |
| `PUT` | `/jobs/{id}` | Update job posting |
| `DELETE` | `/jobs/{id}` | Delete job + chunks |
| `POST` | `/jobs/{id}/upload-jd` | Upload a JD file for a job |
| `POST` | `/jobs/{id}/process` | Trigger JD processing |
| `GET` | `/jobs/{id}/candidates` | Get candidate recommendations for a job |

### Matching

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/match/candidate-to-jobs/{candidate_id}` | Explicitly match candidate → jobs |
| `POST` | `/match/job-to-candidates/{job_id}` | Explicitly match job → candidates |
| `GET` | `/match/explain/{match_id}` | Get detailed explanation for a match |
| `GET` | `/match/history/{candidate_id}` | Get all past matches for a candidate |

### Legacy NexusRAG endpoints

For backward compatibility and chat/analytics functionality, the original NexusRAG endpoints are still available:

- `/workspaces`, `/documents`, `/rag/query`, `/rag/chat/{workspace_id}/stream`, `/rag/graph`, `/rag/analytics`, etc.

---

## Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| **FastAPI** | Async web framework with SSE streaming |
| **SQLAlchemy 2.0** | Async ORM with PostgreSQL (asyncpg) |
| **ChromaDB** | Vector store for CV and JD chunks (`cv_chunks`, `jd_chunks`) |
| **LightRAG** | Knowledge graph (file‑based, per‑workspace) |
| **Docling** | Deep document parsing for CVs and JDs |
| **sentence-transformers** | BAAI/bge‑m3 embeddings + BAAI/bge‑reranker‑v2‑m3 reranking |
| **google‑genai** | Gemini API (chat, extraction, KG embeddings) |
| **ollama** | Local LLM provider (optional) |

### Frontend

| Technology | Purpose |
|---|---|
| **React 19** + **TypeScript 5.9** | UI with strict typing |
| **Vite 7** | Dev server and bundler |
| **TailwindCSS 4** | Styling with dark/light themes |
| **Zustand 5** | Local state management (dashboards, layout) |
| **React Query 5** | Data fetching and caching |
| **Framer Motion 12** | Animations and layout transitions |
| **Lucide React** | Icon set |

### Infrastructure

| Technology | Purpose |
|---|---|
| **PostgreSQL 15** | Relational data (candidates, jobs, matches, chat) |
| **ChromaDB** | Vector embeddings (HTTP client, containerized) |
| **LightRAG** | File‑based KG (NetworkX + NanoVectorDB — no extra services) |
| **Docker Compose** | Full‑stack orchestration (Postgres, ChromaDB, backend, frontend) |
| **nginx** | Production frontend + reverse proxy |

---

<div align="center">

If you find JobNexusRAG (or the underlying NexusRAG pipeline) useful, please consider giving the project a ⭐ — it helps others discover it and motivates further development.

MIT License

Copyright &copy; 2026 Dang Huynh Son
Copyright &copy; 2026 Le Duc Dat

This project is forked and modified from NexusRAG by Le Duc Dat.

</div>

