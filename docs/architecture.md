# SmartEdge Architecture

## Overview

SmartEdge is an ML-driven predictive edge caching system that uses machine learning to make intelligent cache admission decisions. The system combines a Random Forest classifier with a score-based cache eviction policy to outperform traditional caching algorithms like LRU and LFU.

## System Components

### 1. Backend API (FastAPI)

**Location:** `api/main.py`

The FastAPI backend serves as the central API server, providing:

- **Health Check:** `GET /health` - System health and model availability
- **Cache Statistics:** `GET /cache/stats` - Live cache performance metrics
- **Cache Contents:** `GET /cache` - Current cached items
- **Cache Reset:** `POST /cache/reset` - Clear cache and statistics
- **ML Prediction:** `POST /predict` - Manual ML prediction for cache admission
- **Request Processing:** `POST /request` - Simulate a resource request through SmartEdge
- **Benchmark Summary:** `GET /summary` - Offline benchmark comparison results
- **AI Agent:** `POST /agent/ask` - Query the multi-agent AI system

**Configuration:**
- Cache Capacity: 100 items
- Admission Threshold: 0.50 (configurable)
- Model: Random Forest Classifier (5-class classification)

### 2. ML Model

**Location:** `model/smartedge_v3_random_forest.pkl`

**Model Type:** Random Forest Classifier (scikit-learn)

**Features (5):**
1. `hour` - Hour of day (0-23)
2. `day_of_week` - Day of week (0-6, Monday=0)
3. `past_frequency` - Number of previous requests for this content
4. `recency_seconds` - Time since last request in seconds
5. `content_size` - Size of content in bytes

**Prediction Classes (5):**
- 0: NO_REUSE
- 1: LOW_REUSE
- 2: MEDIUM_REUSE
- 3: HIGH_REUSE
- 4: VERY_HIGH_REUSE

**Output:**
- Predicted class
- Class probabilities
- Reusable probability (sum of classes 1-4)
- Expected cache value (weighted sum)

### 3. Cache Engine

**Location:** `api/main.py` (in-memory cache)

**Implementation:** OrderedDict-based cache with ML admission

**Cache Admission Logic:**
1. Extract request features (hour, day_of_week, past_frequency, recency, content_size)
2. Run Random Forest prediction
3. Calculate reusable probability
4. Apply admission threshold (default 0.50)
5. If probability >= threshold:
   - If cache has space: admit immediately
   - If cache is full: compare scores, evict lowest-value item if new item has higher score
6. If probability < threshold: reject (do not cache)

**Cache Score Calculation:**
```
score = expected_value × probability × log(frequency) × (1 / log(age)) × (1 / log(size))
```

**Tracked Metrics:**
- Total requests
- Cache hits
- Cache misses
- Admissions
- Rejections
- Evictions
- ML predictions
- Average response time

### 4. RAG System

**Location:** `rag/retriever.py`

**Technology:**
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector Index: FAISS (IndexFlatL2)

**Knowledge Base:** `rag/documents/`
- `smartedge_overview.txt` - System architecture and ML model details
- `benchmark_results.txt` - LRU/LFU/SmartEdge comparison results
- `adaptive_threshold_results.txt` - Threshold experiment results

**Retrieval Process:**
1. Encode user query using Sentence Transformers
2. Search FAISS index for top-k similar documents
3. Return documents with similarity scores
4. Feed retrieved context to LLM for grounded answers

### 5. Multi-Agent System

**Location:** `agents/`

**Orchestrator:** `agents/orchestrator.py`
- Routes questions to appropriate agent based on keywords
- Provides live SmartEdge context to agents

**Agents:**

**Cache Analyst Agent** (`agents/cache_agent.py`)
- Answers questions about cache decisions, predictions, and reuse
- Uses RAG for SmartEdge-specific knowledge
- Explains ML predictions and cache behavior

**Performance Agent** (`agents/performance_agent.py`)
- Answers performance comparison questions
- Uses structured benchmark data from CSV files
- Explains hit rates, evictions, and algorithm comparisons

**Tuning Agent** (`agents/tuning_agent.py`)
- Provides threshold and capacity recommendations
- Analyzes adaptive threshold experiment results
- Suggests future experiments (advisory only, does not change config)

**LLM Integration:**
- Model: Ollama Llama 3.2 3B
- Purpose: Grounded explanation generation
- Constraint: Agents cannot modify production configuration

### 6. Frontend Dashboard

**Location:** `frontend/`

**Technology:** Vanilla HTML, CSS, JavaScript (no frameworks)

**Features:**
- Real-time KPI cards (requests, hits, misses, cache size)
- ML prediction engine with manual feature input
- Prediction result display with probability gauge
- Cache statistics chart (hit rate over time)
- Cache contents table with remove functionality
- AI assistant interface with evidence display
- Live system context panel
- Analytics page with algorithm comparison
- Request simulator for live cache testing

**API Integration:**
- All API calls to `http://127.0.0.1:8001`
- 5-second auto-refresh for live statistics
- Error handling for backend offline scenarios

## Data Flow

### Request Processing Flow

```
User Request
    ↓
Feature Extraction (hour, day_of_week, past_frequency, recency, content_size)
    ↓
Random Forest Prediction
    ↓
Calculate Reusable Probability
    ↓
Apply Admission Threshold (0.50)
    ↓
Cache Decision (CACHE / DO_NOT_CACHE)
    ↓
Cache Update (if admitted)
    ↓
Statistics Update
    ↓
Response to User
```

### AI Agent Flow

```
User Question
    ↓
Orchestrator Keyword Analysis
    ↓
Agent Selection (Cache / Performance / Tuning)
    ↓
RAG Retrieval (if Cache Agent)
    ↓
Structured Data Query (if Performance/Tuning Agent)
    ↓
LLM Context Construction
    ↓
Llama 3.2 Generation
    ↓
Grounded Response with Evidence
    ↓
Live Context Attachment
    ↓
Response to User
```

## File Structure

```
smart edge/
├── api/
│   ├── main.py              # FastAPI backend with cache logic
│   ├── agent_api.py         # AI agent endpoint
│   ├── cache_engine.py      # Standalone cache implementation
│   └── smartedge_engine.py  # Benchmark simulation engine
├── agents/
│   ├── orchestrator.py      # Multi-agent routing
│   ├── cache_agent.py       # Cache decision analysis
│   ├── performance_agent.py # Performance comparison
│   ├── tuning_agent.py      # Threshold recommendations
│   └── smartedge_tools.py   # Live context tools
├── rag/
│   ├── retriever.py         # FAISS-based retrieval
│   ├── documents/           # Knowledge base text files
│   ├── data/               # Benchmark CSV files
│   └── embeddings/          # FAISS index (generated at runtime)
├── model/
│   ├── smartedge_v3_random_forest.pkl  # Trained ML model
│   └── adaptive_threshold_results_v3.csv # Threshold experiments
├── dataset/
│   └── processed/
│       └── final_ml_dataset.csv  # Training data
├── cache/
│   ├── results/             # Benchmark result charts
│   └── final_cache_comparison.csv # Algorithm comparison
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── script.js            # Frontend logic
│   └── style.css            # Styling
├── docs/                    # Documentation
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
└── README.md                # Project overview
```

## Technology Stack

- **Backend:** FastAPI, Uvicorn, Python 3.8+
- **ML:** scikit-learn, pandas, numpy, joblib
- **RAG:** Sentence Transformers, FAISS
- **LLM:** Ollama, Llama 3.2 3B
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Data:** CSV, JSON
- **Visualization:** Chart.js

## Key Design Decisions

1. **ML-First Caching:** Unlike reactive algorithms (LRU/LFU), SmartEdge predicts future reuse before caching
2. **Deterministic Admission:** Clear threshold-based admission with explainable decisions
3. **Score-Based Eviction:** Multi-factor scoring considers probability, frequency, recency, and size
4. **Grounded AI:** RAG ensures AI answers are based on actual system knowledge, not hallucinations
5. **Advisory Agents:** AI agents provide recommendations but cannot modify production configuration
6. **Real-Time Dashboard:** Live statistics with 5-second refresh for immediate feedback
7. **Research-Ready:** Comprehensive metrics for IEEE paper support
