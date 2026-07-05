# pgvector-tutorial

A hands-on repository for building a RAG system from scratch using pgvector and the Gemini API, covering the full stack from Tool Use, AI Agents, and MCP to cloud deployment and production operations.

## Overview

This repository contains the source code for two Zenn books, also published as article series on Dev.to.

| Book | Topics | Zenn | Dev.to |
|------|--------|------|--------|
| Vol.1 | RAG · Tool Use · AI Agents · MCP · Cloud Deployment | [Zenn](https://zenn.dev/hkame/books/ai-architect-rag) | [Dev.to](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-with-pgvector-and-gemini-introduction-c8i) |
| Vol.2 | Evals · Observability · Security · MLOps · Fine-tuning · Multi-Agent · Governance | [Zenn](https://zenn.dev/hkame/books/ai-architect-production) | [Dev.to](https://dev.to/hiroki-kameyama/taking-rag-to-production-evals-observability-security-and-beyond-introduction-44kb) |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| LLM / Embedding | Google Gemini API (gemini-2.5-flash / gemini-embedding-001) |
| Vector DB | pgvector (PostgreSQL extension) |
| MCP | FastMCP (stdio mode / HTTP mode) |
| Cloud | Render (MCP server) × Supabase (pgvector DB) |
| Observability | Langfuse v4 |
| Fine-tuning | Hugging Face Transformers + PEFT (LoRA) |
| CI/CD | GitHub Actions |
| Language | Python 3.12 |

---

## Directory Structure

```
pgvector-tutorial/
│
├── 01_setup_db.py          # pgvector table and extension setup
├── 02_create_index.py      # HNSW index creation
├── 03_ingest.py            # Document embedding generation and storage
├── 04_search.py            # Vector search and filtered search
├── 05_rag.py               # RAG pipeline
│
├── 06_tool_basic.py        # Tool Use basics (autonomous tool selection)
├── 07_tool_multi.py        # Multiple tool routing
├── 08_tool_agent.py        # Agentic loop with conversation history
│
├── 09_agent_basic.py       # ReAct pattern (3-tool integration)
├── 10_agent_memory.py      # Long-term memory (file persistence)
├── 11_agent_planner.py     # Plan → Execute → Evaluate
│
├── 12_mcp_agent.py         # Agent via MCP client (stdio)
├── 13_mcp_http_agent.py    # Agent via HTTP MCP server
├── 14_multiagent.py        # Multi-agent execution script
│
├── mcp_server/
│   ├── server.py           # MCP server (stdio mode)
│   ├── server_http.py      # MCP server (HTTP mode)
│   └── server_render.py    # MCP server for Render deployment
│
├── evals/
│   ├── dataset.py          # Evaluation dataset (7 test cases)
│   └── eval_rag.py         # Context Recall · Relevancy · Faithfulness
│
├── observability/
│   ├── traced_rag.py       # RAG pipeline tracing with @observe()
│   └── traced_agent.py     # Per-step agent tracing
│
├── security/
│   ├── input_validator.py  # Prompt injection detection
│   ├── output_validator.py # PII masking and leak detection
│   ├── guardrails.py       # Rate limiting and security logging
│   └── secure_rag.py       # RAG with guardrails
│
├── llmops/
│   ├── prompt_registry.py  # Prompt version management (v1.0–v1.2)
│   ├── ci_eval.py          # Quality gate (deploy only if Overall ≥ 75%)
│   └── cost_tracker.py     # API cost tracking
│
├── finetuning/
│   ├── prepare_dataset.py  # Convert to Alpaca format
│   ├── train_lora.py       # Fine-tuning with LoRA (r=8)
│   └── inference.py        # Compare base vs fine-tuned model
│
├── multiagent/
│   ├── search_worker.py    # Specialized search worker
│   ├── quality_worker.py   # Specialized quality check worker
│   └── orchestrator.py     # Task decomposition and result synthesis
│
├── governance/
│   ├── ai_registry.py      # AI system inventory
│   ├── risk_assessor.py    # Risk assessment (EU AI Act compliant)
│   ├── audit_logger.py     # Audit logging (Article 12 compliant)
│   └── compliant_rag.py    # RAG with AI disclosure (Article 50 compliant)
│
└── .github/
    └── workflows/
        └── llmops.yml      # GitHub Actions CI/CD pipeline
```

---

## Setup

### Prerequisites

- Python 3.12 (pyenv recommended)
- Docker
- Google Gemini API key (get one at [AI Studio](https://aistudio.google.com))

### Installation

```bash
git clone https://github.com/qameqame/pgvector-tutorial.git
cd pgvector-tutorial

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with the following:

```
GEMINI_API_KEY=AIza...

# Local DB
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password

# Langfuse (Observability)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Start pgvector

```bash
docker run -d \
  --name pgvector-demo \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Initial Setup

```bash
python 01_setup_db.py
python 02_create_index.py
python 03_ingest.py
```

---

## Usage

### RAG Pipeline

```bash
python 05_rag.py
```

### Start MCP Server (stdio mode)

```bash
python mcp_server/server.py
```

### Multi-Agent System

```bash
python 14_multiagent.py
```

### Quality Evaluation (Evals)

```bash
python evals/eval_rag.py
```

### CI Quality Gate

```bash
python llmops/ci_eval.py
```

### Governance-Compliant RAG

```bash
python governance/compliant_rag.py
```

---

## Cloud Architecture

```
MCP Server  → Render (https://pgvector-tutorial.onrender.com/mcp)
pgvector DB → Supabase (Connection Pooler / port 6543)
```

---

## Design Decisions

### Why pgvector instead of a dedicated vector DB?

Three reasons: it integrates with existing PostgreSQL infrastructure, allows SQL and vector search in the same query, and delivers comparable performance for mid-scale workloads (up to a few million documents).

### Why 768 dimensions for Gemini Embedding?

`gemini-embedding-001` outputs 3072 dimensions by default, but pgvector's HNSW index has a 2000-dimension limit. Setting `output_dimensionality=768` stays well within that limit with negligible impact on retrieval quality.

### Why LoRA rank 8?

`r=8` strikes the right balance — more expressive than `r=4`, more memory-efficient than `r=16`. Training completes in under 2 minutes on CPU.

### EU AI Act compliance scope

This RAG system is classified as **Limited Risk (chatbot)** under the EU AI Act, with a risk score of 0.18 (LOW). Compliance is established by implementing an AI disclosure notice (Article 50) and an audit log (Article 12).

---

## Articles

### Zenn Books
- **Vol.1**: [RAG Implementation Guide for AI Beginners](https://zenn.dev/hkame/books/ai-architect-rag)
- **Vol.2**: [Production Operations Guide for AI Architects](https://zenn.dev/hkame/books/ai-architect-production)

### Dev.to Series
- **Vol.1**: [RAG Implementation Guide for AI Architects](https://dev.to/hiroki-kameyama/building-a-rag-system-from-scratch-with-pgvector-and-gemini-introduction-c8i)
- **Vol.2**: [Production Operations Guide for AI Architects](https://dev.to/hiroki-kameyama/taking-rag-to-production-evals-observability-security-and-beyond-introduction-44kb)

---

## Author

**Hiroki Kameyama**
- Zenn: [@hkame](https://zenn.dev/hkame)
- GitHub: [@qameqame](https://github.com/qameqame)
