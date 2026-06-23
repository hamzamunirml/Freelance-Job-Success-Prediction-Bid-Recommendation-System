# ============================================================
# Shared Test Fixtures
# File: tests/conftest.py
# ============================================================

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from recommendation_engine import BidRecommendationEngine


@pytest.fixture(scope="session")
def test_engine():
    """Create a recommendation engine instance for all tests"""
    try:
        engine = BidRecommendationEngine()
        return engine
    except Exception as e:
        pytest.skip(f"Engine initialization failed: {e}")


@pytest.fixture(scope="session")
def sample_data():
    """Provide sample data for testing"""
    return {
        "project_budget": 5000,
        "client_rating": 4.8,
        "num_existing_proposals": 5,
        "freelancer_experience": 10,
        "proposal_quality_score": 9,
        "project_duration": 30,
        "freelancer_rating": 4.7,
        "previous_jobs_completed": 200,
        "budget_per_proposal": 833.33,
        "project_category_encoded": 0,
        "client_location_encoded": 8,
        "freelancer_country_encoded": 3,
        "client_rating_category_encoded": 0,
        "experience_level_encoded": 1,
        "experience_quality_interaction": 90,
        "combined_experience_rating": 80.5,
        "rating_per_proposal": 0.78,
        "proposal_quality_score_x_freelancer_rating": 42.3,
        "proposal_quality_score_x_freelancer_experience": 90,
        "client_rating_x_freelancer_rating": 22.56,
    }


@pytest.fixture
def sample_dataframe(sample_data):
    """Convert sample data to DataFrame"""
    return pd.DataFrame([sample_data])


@pytest.fixture
def batch_test_data():
    """Generate batch test data"""
    test_cases = []
    for i in range(10):
        test_case = {
            "project_budget": np.random.randint(1000, 50000),
            "client_rating": np.random.uniform(1, 5),
            "num_existing_proposals": np.random.randint(0, 100),
            "freelancer_experience": np.random.randint(0, 20),
            "proposal_quality_score": np.random.randint(1, 10),
            "project_duration": np.random.randint(5, 90),
            "freelancer_rating": np.random.uniform(1, 5),
            "previous_jobs_completed": np.random.randint(0, 500),
            "budget_per_proposal": np.random.randint(10, 10000),
            "project_category_encoded": np.random.randint(0, 6),
            "client_location_encoded": np.random.randint(0, 10),
            "freelancer_country_encoded": np.random.randint(0, 10),
            "client_rating_category_encoded": np.random.randint(0, 4),
            "experience_level_encoded": np.random.randint(0, 4),
            "experience_quality_interaction": np.random.randint(0, 200),
            "combined_experience_rating": np.random.randint(0, 500),
            "rating_per_proposal": np.random.uniform(0, 1),
            "proposal_quality_score_x_freelancer_rating": np.random.uniform(0, 50),
            "proposal_quality_score_x_freelancer_experience": np.random.uniform(0, 200),
            "client_rating_x_freelancer_rating": np.random.uniform(0, 25),
        }
        test_cases.append(test_case)
    return pd.DataFrame(test_cases)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for test files"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)
