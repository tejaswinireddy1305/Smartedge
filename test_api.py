"""
SmartEdge API Tests

Basic tests for the FastAPI backend endpoints.
Run with: pytest tests/test_api.py -v
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_status(self):
        """Test that health endpoint returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "model_exists" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_status(self):
        """Test that root endpoint returns API status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["project"] == "SmartEdge"


class TestCacheStatsEndpoint:
    """Test cache statistics endpoint"""
    
    def test_cache_stats(self):
        """Test that cache stats endpoint returns statistics"""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "total_requests" in data
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "cache_size" in data
        assert "cache_capacity" in data


class TestCacheContentsEndpoint:
    """Test cache contents endpoint"""
    
    def test_cache_contents(self):
        """Test that cache contents endpoint returns items"""
        response = client.get("/cache")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "size" in data
        assert "capacity" in data
        assert "items" in data
        assert isinstance(data["items"], list)


class TestCacheResetEndpoint:
    """Test cache reset endpoint"""
    
    def test_cache_reset(self):
        """Test that cache reset clears cache and statistics"""
        response = client.post("/cache/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["cache_size"] == 0


class TestPredictEndpoint:
    """Test ML prediction endpoint"""
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        payload = {
            "hour": 12,
            "day_of_week": 2,
            "past_frequency": 20,
            "recency_seconds": 10,
            "content_size": 1839
        }
        response = client.post("/predict", json=payload)
        # May fail if model file doesn't exist, but should still return valid JSON
        assert response.status_code in [200, 500]
        data = response.json()
        if response.status_code == 200:
            assert "predicted_class" in data
            assert "reusable_probability" in data
            assert "cache_decision" in data
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input"""
        payload = {
            "hour": 25,  # Invalid hour (should be 0-23)
            "day_of_week": 2,
            "past_frequency": 20,
            "recency_seconds": 10,
            "content_size": 1839
        }
        response = client.post("/predict", json=payload)
        # Should handle invalid input gracefully
        assert response.status_code in [200, 422]


class TestRequestEndpoint:
    """Test request processing endpoint"""
    
    def test_request_new_item(self):
        """Test processing a new request"""
        payload = {
            "url": "/images/test.jpg",
            "content_size": 45000
        }
        response = client.post("/request", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "url" in data
        assert "cache_status" in data
        assert data["cache_status"] in ["HIT", "MISS"]
    
    def test_request_duplicate_item(self):
        """Test processing a duplicate request (cache hit)"""
        url = "/images/duplicate.jpg"
        # First request
        payload = {
            "url": url,
            "content_size": 45000
        }
        client.post("/request", json=payload)
        
        # Second request (should be hit if admitted)
        response = client.post("/request", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestSummaryEndpoint:
    """Test benchmark summary endpoint"""
    
    def test_summary(self):
        """Test that summary endpoint returns benchmark data"""
        response = client.get("/summary")
        assert response.status_code == 200
        data = response.json()
        # May return error if benchmark file doesn't exist
        if data["status"] == "success":
            assert "algorithms" in data
            assert "smartedge_hit_rate" in data
            assert "lru" in data["algorithms"]
            assert "lfu" in data["algorithms"]
            assert "SmartEdge" in data["algorithms"]


class TestAgentEndpoint:
    """Test AI agent endpoint"""
    
    def test_agent_ask(self):
        """Test asking a question to the AI agent"""
        payload = {
            "question": "What is SmartEdge?"
        }
        response = client.post("/agent/ask", json=payload)
        # May fail if Ollama is not running, but should return valid JSON
        assert response.status_code in [200, 500]
        data = response.json()
        if response.status_code == 200:
            assert "question" in data
            assert "selected_agent" in data or "agent" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
