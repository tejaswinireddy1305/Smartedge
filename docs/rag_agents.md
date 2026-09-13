# SmartEdge RAG and AI Agents

## Overview

SmartEdge uses a Retrieval-Augmented Generation (RAG) system combined with a multi-agent architecture to provide intelligent, grounded answers about the caching system. The AI agents use RAG to retrieve relevant knowledge and LLMs to generate explanations.

## RAG System

### Architecture

**Location:** `rag/retriever.py`

**Components:**
1. **Document Store:** Text files in `rag/documents/`
2. **Embedding Model:** Sentence Transformers (all-MiniLM-L6-v2)
3. **Vector Index:** FAISS (IndexFlatL2)
4. **Retriever:** Semantic search over documents

### Knowledge Base

**Location:** `rag/documents/`

**Documents:**

1. **smartedge_overview.txt**
   - System architecture description
   - ML model details
   - Feature descriptions
   - Cache admission logic
   - Comparison with traditional algorithms

2. **benchmark_results.txt**
   - LRU vs LFU vs SmartEdge results
   - Hit rate comparisons
   - Capacity-based performance
   - Improvement metrics

3. **adaptive_threshold_results.txt**
   - Threshold experiment results
   - Capacity vs threshold analysis
   - Admissions and evictions data
   - Performance observations

### Embedding Model

**Model:** all-MiniLM-L6-v2 (Sentence Transformers)

**Dimensions:** 384

**Purpose:** Convert text to vector embeddings for semantic search

**Loading:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Vector Index

**Library:** FAISS (Facebook AI Similarity Search)

**Index Type:** IndexFlatL2 (exact L2 distance search)

**Building:**
```python
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))
```

**Searching:**
```python
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding.astype("float32"), top_k)
```

### Retrieval Process

**Location:** `rag/retriever.py` - `SmartEdgeRetriever.search()`

**Steps:**
1. Encode user query using Sentence Transformers
2. Search FAISS index for top-k similar documents
3. Return documents with similarity scores (distances)

**Parameters:**
- `query`: User question string
- `top_k`: Number of documents to retrieve (default: 2)

**Response:**
```python
[
    {
        "source": "smartedge_overview.txt",
        "text": "SmartEdge is an ML-driven predictive edge caching system...",
        "distance": 0.2345
    }
]
```

## Multi-Agent System

### Orchestrator

**Location:** `agents/orchestrator.py`

**Purpose:** Route questions to appropriate agent based on keywords

**Agent Selection Logic:**

```python
def select_agent(question):
    q = question.lower()
    
    # Tuning Agent keywords
    if any(kw in q for kw in ["threshold", "tune", "recommend", "optimize"]):
        return "tuning"
    
    # Performance Agent keywords
    if any(kw in q for kw in ["compare", "performance", "hit rate", "lfu", "lru"]):
        return "performance"
    
    # Cache Analyst keywords (default)
    return "cache"
```

**Live Context:**
The orchestrator attaches live SmartEdge context to all agent responses:
- Health status
- Cache statistics
- Cache state

### Cache Analyst Agent

**Location:** `agents/cache_agent.py`

**Purpose:** Answer questions about cache decisions, predictions, and system behavior

**Questions Handled:**
- "Why was this request rejected?"
- "Why was this content cached?"
- "What does the prediction probability mean?"
- "How does SmartEdge make a cache decision?"

**Process:**
1. Retrieve relevant documents using RAG
2. Build context from retrieved evidence
3. Send context to LLM with question
4. Return grounded answer with evidence

**Prompt Template:**
```
You are the SmartEdge Cache Analyst.

SmartEdge is an ML-driven predictive edge caching system.

Answer the user's question using ONLY the evidence provided below.

Do not invent benchmark numbers.

If the evidence does not contain enough information, clearly say that the available evidence is insufficient.

USER QUESTION: {question}

RETRIEVED EVIDENCE: {context}

Provide a concise answer.
```

**Response:**
```python
{
    "question": "Why was this request rejected?",
    "answer": "The request was rejected because the predicted reusable probability...",
    "evidence": [
        {"source": "smartedge_overview.txt", "distance": 0.2345}
    ]
}
```

### Performance Agent

**Location:** `agents/performance_agent.py`

**Purpose:** Answer performance comparison questions using structured benchmark data

**Questions Handled:**
- "Compare SmartEdge with LFU."
- "Which algorithm has the best hit rate?"
- "How does cache capacity affect performance?"
- "Explain the benchmark."

**Data Source:** `rag/data/final_cache_comparison.csv`

**Process:**
1. Load benchmark CSV
2. Extract LRU, LFU, SmartEdge results
3. Calculate comparisons and improvements
4. Build findings summary
5. Send findings to LLM for explanation
6. Return grounded analysis

**Findings Generated:**
- SmartEdge hit rate
- LRU hit rate
- LFU hit rate
- SmartEdge vs LRU improvement
- SmartEdge vs LFU improvement
- Hits, misses, evictions

**Prompt Template:**
```
You are the SmartEdge Performance Analyst.

Analyze the benchmark information below.

IMPORTANT:
Use ONLY the supplied numerical evidence.
Do not invent numbers.
Do not change any numerical values.

Explain the performance clearly for a technical project report.

USER QUESTION: {question}

BENCHMARK EVIDENCE: {findings_text}

Give:
1. Direct answer
2. Important numerical comparison
3. Short interpretation
```

**Response:**
```python
{
    "question": "Compare SmartEdge with LFU.",
    "answer": "SmartEdge achieves a 2.26 percentage point improvement over LFU...",
    "benchmark_source": "final_cache_comparison.csv",
    "findings": [
        "SmartEdge hit rate: 68.11%",
        "LFU hit rate: 65.85%",
        "SmartEdge difference versus LFU: 2.26 percentage points"
    ]
}
```

### Tuning Agent

**Location:** `agents/tuning_agent.py`

**Purpose:** Provide threshold and capacity recommendations based on experimental data

**Questions Handled:**
- "What threshold should I test?"
- "What experiment should I run next?"
- "How does threshold affect performance?"
- "Which capacity performed best?"

**Data Source:** `rag/data/adaptive_threshold_results_v3.csv`

**Process:**
1. Load adaptive threshold CSV
2. Find best overall result
3. Extract best threshold by capacity
4. Analyze threshold behavior at capacity 100
5. Build evidence summary
6. Send evidence to LLM for recommendations
7. Return grounded recommendations

**Evidence Generated:**
- Best overall SmartEdge hit rate
- Best threshold by capacity
- Threshold behavior at capacity 100
- Admissions and evictions data
- Improvement metrics

**Prompt Template:**
```
You are the SmartEdge Tuning Agent.

Your task is to analyze historical SmartEdge threshold experiments and recommend useful future experiments.

IMPORTANT RULES:

1. Use ONLY the supplied experimental evidence.
2. Never invent benchmark numbers.
3. Clearly distinguish observed results from recommendations.
4. Do NOT claim that a proposed threshold has already been validated.
5. Do NOT change SmartEdge configuration.
6. Recommendations must be tested through a benchmark before adoption.

USER QUESTION: {question}

EXPERIMENTAL EVIDENCE: {evidence_text}

Answer using this structure:

OBSERVED RESULT:
State what the experiments actually show.

RECOMMENDATION:
Suggest the next useful experiment.

REASON:
Explain why that experiment is useful.

CAUTION:
Explain that the recommendation must be benchmarked before being used in production.
```

**Response:**
```python
{
    "question": "What threshold should I test?",
    "recommendation": "Based on the experiments, testing threshold 0.15 at capacity 100...",
    "evidence": [
        "Best overall SmartEdge hit rate: 68.11%",
        "Capacity: 100",
        "Threshold: 0.1"
    ]
}
```

**Important:** The Tuning Agent provides recommendations only. It cannot modify production configuration.

## LLM Integration

### Ollama Setup

**Model:** Llama 3.2 3B

**Purpose:** Generate grounded explanations from retrieved evidence

**Installation:**
```bash
# Install Ollama
# Download from https://ollama.ai

# Pull the model
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

### Python Client

**Library:** ollama Python package

**Usage:**
```python
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]
```

### Model Selection

**Why Llama 3.2 3B?**
- Small enough for local deployment
- Good reasoning capabilities
- Fast inference time
- Low resource requirements
- Open source

**Alternatives:**
- Llama 3.2 1B (faster, less capable)
- Mistral 7B (more capable, slower)
- GPT-4 (cloud-based, requires API key)

## Agent API

### Endpoint

**POST /agent/ask**

**Request:**
```json
{
  "question": "Why does SmartEdge perform better than LFU?"
}
```

**Response:**
```json
{
  "question": "Why does SmartEdge perform better than LFU?",
  "selected_agent": "Performance Agent",
  "result": {
    "question": "Why does SmartEdge perform better than LFU?",
    "answer": "SmartEdge achieves a 2.26 percentage point improvement...",
    "benchmark_source": "final_cache_comparison.csv",
    "findings": [...]
  },
  "live_context": {
    "health": {...},
    "cache_stats": {...},
    "cache_state": {...}
  }
}
```

## Live Context

**Location:** `agents/smartedge_tools.py`

**Purpose:** Provide real-time SmartEdge state to agents

**Context Components:**
- Health status (model availability, cache capacity)
- Cache statistics (hits, misses, hit rate)
- Cache state (current items, utilization)

**Usage:**
```python
from agents.smartedge_tools import get_live_context

context = get_live_context()
# Returns dict with health, cache_stats, cache_state
```

## Grounding and Safety

### Grounding Principles

1. **Evidence-Based:** All answers must be based on retrieved evidence or structured data
2. **No Hallucination:** Agents must not invent benchmark numbers or system behavior
3. **Source Attribution:** All answers must include source references
4. **Uncertainty Handling:** If evidence is insufficient, clearly state limitations

### Safety Constraints

1. **No Configuration Changes:** Agents cannot modify production threshold or capacity
2. **Advisory Only:** Tuning Agent provides recommendations, not commands
3. **Human Oversight:** All recommendations require human validation before implementation
4. **Explainable:** All decisions must be explainable through evidence

### Error Handling

If RAG fails:
- Return "No relevant SmartEdge evidence was found"
- Agent cannot provide grounded answer
- Suggest checking knowledge base

If Ollama is unavailable:
- Return "AI service unavailable"
- Suggest checking Ollama server status
- Rest of system continues functioning

## Performance Considerations

### RAG Latency

- **Embedding Generation:** ~10-20ms per query
- **FAISS Search:** ~1-5ms for top-2 results
- **Total RAG Latency:** ~15-25ms

### LLM Latency

- **Llama 3.2 3B Inference:** ~100-500ms depending on prompt length
- **Total Agent Latency:** ~150-550ms (RAG + LLM)

### Optimization Opportunities

1. **Cache Embeddings:** Pre-compute and cache document embeddings
2. **Batch Retrieval:** Retrieve multiple documents in parallel
3. **Model Quantization:** Use quantized LLM for faster inference
4. **Response Caching:** Cache common question-answer pairs
5. **Async Processing:** Run RAG and LLM asynchronously

## Testing

### Test RAG Retrieval

```bash
cd "D:\smart edge"
python -c "from rag.retriever import SmartEdgeRetriever; r = SmartEdgeRetriever(); print(r.search('Why does SmartEdge perform better than LFU?'))"
```

### Test Individual Agents

```bash
# Cache Analyst
python agents/cache_agent.py

# Performance Agent
python agents/performance_agent.py

# Tuning Agent
python agents/tuning_agent.py

# Orchestrator
python agents/orchestrator.py
```

### Test Agent API

```bash
curl -X POST http://127.0.0.1:8001/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why does SmartEdge perform better than LFU?"}'
```

## Troubleshooting

### RAG Issues

**Problem:** No documents retrieved
**Solution:** Check that documents exist in `rag/documents/`

**Problem:** FAISS index not built
**Solution:** Ensure Sentence Transformers is installed and documents are valid

### Agent Issues

**Problem:** Ollama connection failed
**Solution:** 
1. Check Ollama is running: `ollama serve`
2. Check model is pulled: `ollama pull llama3.2:3b`
3. Check Python ollama package is installed

**Problem:** Agent returns generic response
**Solution:** Check that relevant documents exist in knowledge base

### Context Issues

**Problem:** Live context shows errors
**Solution:** Check that backend API is running at `http://127.0.0.1:8001`
