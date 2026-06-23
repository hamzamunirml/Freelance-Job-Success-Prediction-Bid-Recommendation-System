# ============================================================
# Unit Tests for Recommendation Engine
# File: tests/test_recommendation_engine.py
# ============================================================

import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from recommendation_engine import BidRecommendationEngine

# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def engine():
    """Create a recommendation engine instance for testing"""
    return BidRecommendationEngine(
        model_path="models/best_model.pkl",
        scaler_path="models/scaler.pkl",
        threshold_path="models/best_threshold.pkl",
    )


@pytest.fixture
def sample_high_prob_input():
    """Sample input for high probability project"""
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
def sample_low_prob_input():
    """Sample input for low probability project"""
    return {
        "project_budget": 50000,
        "client_rating": 1.5,
        "num_existing_proposals": 95,
        "freelancer_experience": 2,
        "proposal_quality_score": 3,
        "project_duration": 45,
        "freelancer_rating": 2.1,
        "previous_jobs_completed": 10,
        "budget_per_proposal": 520.83,
        "project_category_encoded": 5,
        "client_location_encoded": 2,
        "freelancer_country_encoded": 4,
        "client_rating_category_encoded": 1,
        "experience_level_encoded": 2,
        "experience_quality_interaction": 6,
        "combined_experience_rating": 15.5,
        "rating_per_proposal": 0.02,
        "proposal_quality_score_x_freelancer_rating": 6.3,
        "proposal_quality_score_x_freelancer_experience": 6,
        "client_rating_x_freelancer_rating": 3.15,
    }


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing"""
    return pd.DataFrame(
        [
            {
                "project_budget": 8000,
                "client_rating": 4.5,
                "num_existing_proposals": 12,
                "freelancer_experience": 15,
                "proposal_quality_score": 8,
                "project_duration": 25,
                "freelancer_rating": 4.6,
                "previous_jobs_completed": 350,
                "budget_per_proposal": 615.38,
                "project_category_encoded": 0,
                "client_location_encoded": 8,
                "freelancer_country_encoded": 5,
                "client_rating_category_encoded": 0,
                "experience_level_encoded": 1,
                "experience_quality_interaction": 120,
                "combined_experience_rating": 115.3,
                "rating_per_proposal": 0.35,
                "proposal_quality_score_x_freelancer_rating": 36.8,
                "proposal_quality_score_x_freelancer_experience": 120,
                "client_rating_x_freelancer_rating": 20.7,
            },
            {
                "project_budget": 15000,
                "client_rating": 2.5,
                "num_existing_proposals": 75,
                "freelancer_experience": 3,
                "proposal_quality_score": 4,
                "project_duration": 45,
                "freelancer_rating": 2.8,
                "previous_jobs_completed": 25,
                "budget_per_proposal": 197.37,
                "project_category_encoded": 5,
                "client_location_encoded": 2,
                "freelancer_country_encoded": 4,
                "client_rating_category_encoded": 1,
                "experience_level_encoded": 2,
                "experience_quality_interaction": 12,
                "combined_experience_rating": 20.5,
                "rating_per_proposal": 0.04,
                "proposal_quality_score_x_freelancer_rating": 11.2,
                "proposal_quality_score_x_freelancer_experience": 12,
                "client_rating_x_freelancer_rating": 7.0,
            },
        ]
    )


# ============================================================
# Test Class: TestBidRecommendationEngine
# ============================================================


class TestBidRecommendationEngine:
    """Test suite for BidRecommendationEngine"""

    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly"""
        assert engine is not None
        assert engine.model is not None
        assert engine.scaler is not None
        assert engine.threshold is not None
        assert isinstance(engine.feature_columns, list)
        assert len(engine.feature_columns) > 0

    def test_feature_columns(self, engine):
        """Test that feature columns match expected training features"""
        expected_features = [
            "project_budget",
            "client_rating",
            "num_existing_proposals",
            "freelancer_experience",
            "proposal_quality_score",
            "project_duration",
            "freelancer_rating",
            "previous_jobs_completed",
            "budget_per_proposal",
            "project_category_encoded",
            "client_location_encoded",
            "freelancer_country_encoded",
            "client_rating_category_encoded",
            "experience_level_encoded",
            "experience_quality_interaction",
            "combined_experience_rating",
        ]

        # Check that essential features exist
        for feature in expected_features:
            assert feature in engine.feature_columns, f"Missing feature: {feature}"

    def test_prepare_features_with_dict(self, engine, sample_high_prob_input):
        """Test feature preparation with dictionary input"""
        X_scaled = engine.prepare_features(sample_high_prob_input)
        assert X_scaled is not None
        assert isinstance(X_scaled, np.ndarray)
        assert X_scaled.shape[1] == len(engine.feature_columns)
        assert X_scaled.shape[0] == 1

    def test_prepare_features_with_dataframe(self, engine, sample_high_prob_input):
        """Test feature preparation with DataFrame input"""
        df = pd.DataFrame([sample_high_prob_input])
        X_scaled = engine.prepare_features(df)
        assert X_scaled is not None
        assert isinstance(X_scaled, np.ndarray)
        assert X_scaled.shape[1] == len(engine.feature_columns)
        assert X_scaled.shape[0] == 1

    def test_prepare_features_missing_columns(self, engine):
        """Test feature preparation with missing columns"""
        incomplete_input = {
            "project_budget": 5000,
            "client_rating": 4.8,
            "num_existing_proposals": 5,
        }
        X_scaled = engine.prepare_features(incomplete_input)
        assert X_scaled is not None
        assert X_scaled.shape[1] == len(engine.feature_columns)

    def test_predict_success_probability_high(self, engine, sample_high_prob_input):
        """Test prediction for high probability project"""
        probability = engine.predict_success_probability(sample_high_prob_input)
        assert probability is not None
        assert 0 <= probability <= 1
        assert probability >= 0.70  # Should be high probability

    def test_predict_success_probability_low(self, engine, sample_low_prob_input):
        """Test prediction for low probability project"""
        probability = engine.predict_success_probability(sample_low_prob_input)
        assert probability is not None
        assert 0 <= probability <= 1
        assert probability <= 0.40  # Should be low probability

    def test_get_recommendation_high(self, engine):
        """Test recommendation for high probability"""
        recommendation = engine.get_recommendation(0.85)
        assert recommendation["status"] == "HIGH_PROBABILITY"
        assert recommendation["level"] == "High"
        assert recommendation["action"] == "Apply Immediately"
        assert len(recommendation["recommendations"]) > 0
        assert recommendation["color"] == "#2ecc71"

    def test_get_recommendation_medium(self, engine):
        """Test recommendation for medium probability"""
        recommendation = engine.get_recommendation(0.55)
        assert recommendation["status"] == "MEDIUM_PROBABILITY"
        assert recommendation["level"] == "Medium"
        assert recommendation["action"] == "Apply with Caution"
        assert len(recommendation["recommendations"]) > 0
        assert recommendation["color"] == "#f39c12"

    def test_get_recommendation_low(self, engine):
        """Test recommendation for low probability"""
        recommendation = engine.get_recommendation(0.25)
        assert recommendation["status"] == "LOW_PROBABILITY"
        assert recommendation["level"] == "Low"
        assert recommendation["action"] == "Avoid Applying"
        assert len(recommendation["recommendations"]) > 0
        assert recommendation["color"] == "#e74c3c"

    def test_get_recommendation_edge_cases(self, engine):
        """Test recommendation at boundary values"""
        # Test exactly at 0.70
        rec_high = engine.get_recommendation(0.70)
        assert rec_high["status"] == "HIGH_PROBABILITY"

        # Test exactly at 0.40
        rec_medium = engine.get_recommendation(0.40)
        assert rec_medium["status"] == "MEDIUM_PROBABILITY"

        # Test below 0.40
        rec_low = engine.get_recommendation(0.39)
        assert rec_low["status"] == "LOW_PROBABILITY"

    def test_get_full_recommendation(self, engine, sample_high_prob_input):
        """Test full recommendation generation"""
        result = engine.get_full_recommendation(sample_high_prob_input)
        assert result is not None
        assert "probability" in result
        assert "status" in result
        assert "action" in result
        assert "recommendations" in result
        assert "threshold" in result
        assert "decision" in result
        assert "timestamp" in result

    def test_batch_recommendations(self, engine, sample_batch_data):
        """Test batch recommendations"""
        results = engine.batch_recommendations(sample_batch_data)
        assert results is not None
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(sample_batch_data)
        assert "project_id" in results.columns
        assert "probability" in results.columns
        assert "status" in results.columns
        assert "level" in results.columns
        assert "action" in results.columns

    def test_generate_report(self, engine, sample_high_prob_input, tmp_path):
        """Test report generation"""
        output_file = tmp_path / "test_report.json"
        report = engine.generate_report(sample_high_prob_input, str(output_file))
        assert report is not None
        assert os.path.exists(output_file)

        # Verify JSON content
        with open(output_file, "r") as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "recommendation" in data
        assert "input_data" in data
        assert "model_info" in data

    def test_get_decision_matrix(self, engine):
        """Test decision matrix"""
        matrix = engine.get_decision_matrix()
        assert matrix is not None
        assert "High (≥70%)" in matrix
        assert "Medium (40-69%)" in matrix
        assert "Low (<40%)" in matrix

        high = matrix["High (≥70%)"]
        assert "action" in high
        assert "strategy" in high
        assert "time_investment" in high
        assert "win_probability" in high

    def test_engine_predicts_consistently(self, engine, sample_high_prob_input):
        """Test that predictions are consistent"""
        prob1 = engine.predict_success_probability(sample_high_prob_input)
        prob2 = engine.predict_success_probability(sample_high_prob_input)
        assert prob1 == prob2

    def test_probability_range(self, engine, sample_high_prob_input):
        """Test that probability is always between 0 and 1"""
        probability = engine.predict_success_probability(sample_high_prob_input)
        assert 0 <= probability <= 1

    def test_model_file_exists(self):
        """Test that model files exist"""
        assert os.path.exists("models/best_model.pkl"), "best_model.pkl not found"
        assert os.path.exists("models/scaler.pkl"), "scaler.pkl not found"
        assert os.path.exists(
            "models/best_threshold.pkl"
        ), "best_threshold.pkl not found"


# ============================================================
# Test Class: TestIntegration
# ============================================================


class TestIntegration:
    """Integration tests for the recommendation system"""

    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        engine = BidRecommendationEngine()

        input_data = {
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

        result = engine.get_full_recommendation(input_data)

        assert result["probability"] >= 0.70
        assert result["status"] == "HIGH_PROBABILITY"
        assert result["decision"] == "PROCEED"

    def test_batch_integration(self):
        """Test batch processing integration"""
        engine = BidRecommendationEngine()

        # Create multiple test cases
        test_cases = []
        for i in range(5):
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
                "proposal_quality_score_x_freelancer_experience": np.random.uniform(
                    0, 200
                ),
                "client_rating_x_freelancer_rating": np.random.uniform(0, 25),
            }
            test_cases.append(test_case)

        df = pd.DataFrame(test_cases)
        results = engine.batch_recommendations(df)

        assert len(results) == len(test_cases)
        assert all(0 <= prob <= 1 for prob in results["probability"])


# ============================================================
# Test Class: TestPerformance
# ============================================================


class TestPerformance:
    """Performance tests for the recommendation engine"""

    def test_prediction_time(self, engine, sample_high_prob_input):
        """Test that prediction is fast (< 100ms)"""
        import time

        start = time.time()
        for _ in range(100):
            engine.predict_success_probability(sample_high_prob_input)
        end = time.time()

        avg_time = (end - start) / 100
        assert (
            avg_time < 0.1
        ), f"Average prediction time: {avg_time:.3f}s (should be < 0.1s)"

    def test_batch_prediction_time(self, engine):
        """Test that batch prediction is efficient"""
        import time

        # Create 100 test cases
        test_cases = []
        for i in range(100):
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
                "proposal_quality_score_x_freelancer_experience": np.random.uniform(
                    0, 200
                ),
                "client_rating_x_freelancer_rating": np.random.uniform(0, 25),
            }
            test_cases.append(test_case)

        df = pd.DataFrame(test_cases)

        start = time.time()
        results = engine.batch_recommendations(df)
        end = time.time()

        total_time = end - start
        assert (
            total_time < 1.0
        ), f"Batch prediction time: {total_time:.2f}s (should be < 1.0s)"


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
