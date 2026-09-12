# SmartEdge - ML-Driven Predictive Edge Caching

An advanced edge caching system that uses machine learning to predict content reuse and make intelligent cache admission decisions. SmartEdge outperforms traditional caching algorithms (LRU, LFU) by proactively caching resources with high reuse potential.

## Features

- **ML-Powered Caching:** Random Forest classifier predicts content reuse behavior
- **Multi-Agent AI System:** RAG-based agents for cache analysis, performance comparison, and tuning recommendations
- **Real-Time Dashboard:** Professional web interface with live statistics and ML prediction engine
- **Benchmark Suite:** Comprehensive comparison with LRU and LFU algorithms
- **Adaptive Thresholds:** Configurable admission thresholds for different scenarios
- **Explainable Decisions:** Every cache decision includes probability and expected value
- **Grounded AI:** RAG ensures AI answers are based on actual system knowledge
- **Fallback Mode:** System works without Ollama using TF-IDF for RAG and direct evidence display

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **scikit-learn** - Machine learning library
- **pandas/numpy** - Data processing
- **Sentence Transformers** - Text embeddings for RAG (with TF-IDF fallback)
- **FAISS** - Vector similarity search (optional, with TF-IDF fallback)
- **Ollama** - Local LLM (Llama 3.2 3B) - optional for AI agents

### Frontend
- **HTML5/CSS3/JavaScript** - Vanilla web technologies (no frameworks)
- **Chart.js** - Data visualization

### ML Model
- **Random Forest Classifier** - 5-class reuse prediction
- **Features:** hour, day_of_week, past_frequency, recency_seconds, content_size

## Project Structure

```
smart edge/
├── api/                    # FastAPI backend
│   ├── main.py            # Main API server
│   ├── agent_api.py       # AI agent endpoint (with lazy initialization)
│   ├── cache_engine.py    # Cache implementation
│   └── smartedge_engine.py # Benchmark simulation
├── agents/                # AI agents (with Ollama fallback)
│   ├── orchestrator.py    # Multi-agent routing
│   ├── cache_agent.py     # Cache decision analysis
│   ├── performance_agent.py # Performance comparison
│   ├── tuning_agent.py    # Threshold recommendations
│   └── smartedge_tools.py # Live context tools
├── rag/                   # RAG system (with TF-IDF fallback)
│   ├── retriever.py       # FAISS/TF-IDF-based retrieval
│   ├── documents/         # Knowledge base
│   └── data/             # Benchmark CSV files
├── model/                 # ML model and results
│   ├── smartedge_v3_random_forest.pkl
│   └── adaptive_threshold_results_v3.csv
├── dataset/               # Training data
│   └── processed/
│       └── final_ml_dataset.csv
├── cache/                 # Benchmark results
│   ├── results/
│   └── final_cache_comparison.csv
├── frontend/              # Web dashboard
│   ├── index.html
│   ├── script.js
│   └── style.css
├── docs/                  # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── ml_pipeline.md
│   ├── caching.md
│   ├── rag_agents.md
│   └── benchmarking.md
├── tests/                 # Test suite
│   └── test_api.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher (tested with Python 3.13)
- Ollama (optional, for AI agents with LLM)
- Git (optional, for cloning)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd "smart edge"

# Or download and extract the zip file
```

### Step 2: Install Python Dependencies

```bash
cd "D:\smart edge"
pip install -r requirements.txt
```

### Step 3: Install and Configure Ollama (Optional)

1. Download Ollama from https://ollama.ai
2. Install Ollama
3. Pull the required model:

```bash
ollama pull llama3.2:3b
```

4. Start Ollama server (it will run in background):

```bash
ollama serve
```

**Note:** Ollama is optional. The system will work without Ollama using fallback mode (TF-IDF for RAG, direct evidence display for agents). AI features will be more limited without Ollama.

### Step 4: Verify Model and Dataset

Ensure the following files exist:
- `model/smartedge_v3_random_forest.pkl` (~690MB)
- `dataset/processed/final_ml_dataset.csv` (~115MB)
- `cache/results/final_cache_comparison.csv`

If these files are missing, the system will still run but ML prediction and benchmark features will be limited.

## Running the System

### Start the Backend API

**Using Python 3.13 (recommended):**

```bash
cd "D:\smart edge"
py -3.13 -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

**Using default Python:**

```bash
cd "D:\smart edge"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

The API will start at `http://127.0.0.1:8001`

You should see:
```
SMARTEDGE 3.0 API
======================================================================
Model: model/smartedge_v3_random_forest.pkl
Model exists: True
Dataset exists: True
Cache capacity: 100
Threshold: 0.50
======================================================================
```

### Start the Frontend

**Using Python 3.13 (recommended):**

```bash
cd "D:\smart edge\frontend"
py -3.13 -m http.server 5500
```

**Using default Python:**

```bash
cd "D:\smart edge\frontend"
python -m http.server 5500
```

Then open in browser: `http://127.0.0.1:5500`

The dashboard will connect to the backend API at `http://127.0.0.1:8001`

## Testing

### Test Backend Health

```bash
curl http://127.0.0.1:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "SmartEdge 3.0",
  "model_exists": true,
  "dataset_exists": true,
  "results_exists": true,
  "cache_capacity": 100,
  "current_cache_size": 0
}
```

### Test ML Prediction

```bash
curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d "{\"hour\":12,\"day_of_week\":2,\"past_frequency\":20,\"recency_seconds\":10,\"content_size\":1839}"
```

### Test Cache Request

```bash
curl -X POST http://127.0.0.1:8001/request -H "Content-Type: application/json" -d "{\"url\":\"/images/example.jpg\",\"content_size\":45000}"
```

### Test AI Agent (works with or without Ollama)

```bash
curl -X POST http://127.0.0.1:8001/agent/ask -H "Content-Type: application/json" -d "{\"question\":\"Why does SmartEdge perform better than LFU?\"}"
```

### Test RAG Retrieval

```bash
cd "D:\smart edge"
py -3.13 -c "from rag.retriever import SmartEdgeRetriever; r = SmartEdgeRetriever(); print(r.search('Why does SmartEdge perform better than LFU?'))"
```

## API Endpoints

### Core Endpoints

- `GET /` - API status
- `GET /health` - System health check
- `GET /cache/stats` - Live cache statistics
- `GET /cache` - Current cache contents
- `DELETE /cache/item?url={url}` - Remove cache item
- `POST /cache/reset` - Reset cache and statistics
- `POST /predict` - ML prediction for cache admission
- `POST /request` - Process a resource request
- `GET /summary` - Benchmark summary
- `POST /agent/ask` - Query AI agents

For detailed API documentation, see [docs/api.md](docs/api.md)

## Documentation

- [Architecture](docs/architecture.md) - System architecture and components
- [API Documentation](docs/api.md) - Complete API reference
- [ML Pipeline](docs/ml_pipeline.md) - Machine learning model details
- [Caching System](docs/caching.md) - Cache implementation and algorithms
- [RAG and Agents](docs/rag_agents.md) - RAG system and multi-agent architecture
- [Benchmarking](docs/benchmarking.md) - Benchmark methodology and results

## Benchmark Results

### NASA Trace (10,000 requests, capacity 100)

| Algorithm | Hit Rate | Evictions |
|-----------|----------|-----------|
| LRU       | 59.54%   | 36,400    |
| LFU       | 65.85%   | 32,193    |
| SmartEdge | 68.11%   | 1,227     |

**SmartEdge Improvement:**
- +8.57 percentage points vs LRU
- +2.26 percentage points vs LFU
- 96.6% fewer evictions than LRU

For detailed benchmark results, see [docs/benchmarking.md](docs/benchmarking.md)

## Configuration

### Cache Configuration

Edit `api/main.py` to change:

```python
CACHE_CAPACITY = 100  # Number of items in cache
THRESHOLD = 0.50      # Admission threshold (0.0 - 1.0)
```

### Backend URL

Edit `frontend/script.js` to change:

```javascript
const API_URL = "http://127.0.0.1:8001";
```

### Ollama Model

Edit agent files to use a different Ollama model:

```python
# In agents/cache_agent.py, performance_agent.py, tuning_agent.py
response = ollama.chat(model="llama3.2:3b", messages=[...])
```

## Troubleshooting

### Backend fails to start

**Problem:** Model file not found
**Solution:** Ensure `model/smartedge_v3_random_forest.pkl` exists

**Problem:** Port 8001 already in use
**Solution:** Change port in uvicorn command:
```bash
py -3.13 -m uvicorn api.main:app --host 127.0.0.1 --port 8002
```

**Problem:** Python version mismatch
**Solution:** Use `py -3.13` instead of `python` to ensure correct Python version

### AI agent returns errors

**Problem:** Ollama connection failed
**Solution:**
1. Check Ollama is running: `ollama serve`
2. Check model is pulled: `ollama pull llama3.2:3b`
3. Check Python ollama package is installed
4. **Fallback:** System will work without Ollama using direct evidence display

**Problem:** Out of memory error with Ollama
**Solution:** System automatically falls back to non-LLM mode. No action needed.

### Frontend cannot connect to backend

**Problem:** CORS error or connection refused
**Solution:**
1. Ensure backend is running at `http://127.0.0.1:8001`
2. Check API_URL in `frontend/script.js` matches backend URL
3. Backend CORS is enabled for all origins

### ML prediction fails

**Problem:** Invalid input parameters
**Solution:** Ensure all 5 features are provided:
- hour (0-23)
- day_of_week (0-6)
- past_frequency (integer >= 0)
- recency_seconds (integer >= 0)
- content_size (integer >= 0)

### RAG fails to load Sentence Transformer

**Problem:** SSL certificate error or memory error
**Solution:** System automatically falls back to TF-IDF. No action needed.

## Development

### Running Tests

```bash
cd "D:\smart edge"
py -3.13 -m pytest tests/ -v
```

### Running Benchmarks

```bash
# Complete algorithm comparison
py -3.13 api/smartedge_engine.py

# Adaptive threshold experiment
py -3.13 api/adaptive_threshold_experiment_v3.py

# Capacity experiment
py -3.13 api/capacity_experiment_v3.py
```

### Adding New Agents

1. Create new agent file in `agents/`
2. Implement agent class with `analyze()` method
3. Add Ollama availability check in `__init__`
4. Add routing logic to `agents/orchestrator.py`
5. Register agent in `api/agent_api.py` if needed

## Contributing

This is a research project. Contributions should focus on:
- Improving ML model accuracy
- Adding new caching algorithms for comparison
- Enhancing RAG knowledge base
- Improving AI agent capabilities
- Adding comprehensive tests

## License

This project is for research and educational purposes.

## Citation

If you use SmartEdge in your research, please cite:

```
SmartEdge: ML-Driven Predictive Edge Caching for Content Delivery Networks
[Author Names], [Year]
IEEE/ACM Transactions on Networking
```

## Contact

For questions or issues, please refer to the documentation in the `docs/` directory.

## Acknowledgments

- NASA for providing web server access logs
- scikit-learn for ML framework
- Sentence Transformers for embeddings
- FAISS for vector search
- Ollama for local LLM inference
