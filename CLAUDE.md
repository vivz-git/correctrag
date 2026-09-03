# CorrectRAG

## 1. Project Goal

CorrectRAG is a serious but simple, explainable GenAI portfolio project.
It is a lightweight engineering adaptation of the Corrective Retrieval
Augmented Generation (CRAG) framework. It evaluates retrieved evidence,
routes the query through CORRECT, AMBIGUOUS, or INCORRECT paths, and uses
web search when internal knowledge is insufficient.

## 2. Current Architecture

Vercel frontend
→ HTTPS
→ Caddy
→ FastAPI
→ Gemini embeddings
→ in-memory vector retrieval
→ similarity pre-filter
→ Groq LLM judge for borderline retrievals
→ CORRECT / AMBIGUOUS / INCORRECT
→ knowledge refinement / query rewriting / Tavily web fallback
→ grounded generation

Backend:
- Python 3.10+
- FastAPI
- Pydantic

Retrieval:
- Gemini Embedding API
- pure-Python in-memory vector store
- cosine similarity

LLMs:
- Groq for borderline relevance judging
- Gemini/Groq/OpenAI for final generation where configured

External search:
- Tavily API

Frontend:
- plain HTML/CSS/JavaScript

## 3. Frozen Constraints

- Keep the project simple and explainable.
- Prefer the smallest correct change.
- Do not perform unrelated refactors.
- Do not reintroduce PyTorch.
- Do not reintroduce SentenceTransformers.
- Do not reintroduce CrossEncoder.
- Do not reintroduce ChromaDB.
- Do not redesign the frontend.
- Do not add authentication or user accounts.
- Do not add Redis, RDS, ECS, load balancers, or unnecessary infrastructure.
- Never expose API keys or .env contents.
- Never weaken or delete tests just to make them pass.
- Do not change the architecture unless explicitly requested.

## 4. Tests

Run:

pytest -q tests

Run the test suite after meaningful code changes.
Fix underlying failures instead of weakening or deleting tests.

## 5. Deployment Facts

- AWS EC2: t4g.micro
- Region: ap-south-1
- OS: Amazon Linux 2023 ARM64
- Docker runs the FastAPI backend.
- Caddy is the reverse proxy and TLS terminator.
- FastAPI listens privately on 127.0.0.1:8000.
- Public backend uses HTTPS.
- Vercel hosts the frontend.
- Vercel connects to the AWS HTTPS backend.

## 6. Rules for Future Changes

- Inspect relevant files before editing.
- Change only what the task requires.
- Prefer small, general fixes over query-specific hacks.
- Do not modify unrelated files.
- Run tests after meaningful changes.
- Inspect git diff before committing.
- Never commit or push unless explicitly instructed.
- Keep the system easy for a fresher GenAI engineer to explain end-to-end.
