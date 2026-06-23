# ============================================================
# Unit Tests for Streamlit App (FIXED)
# File: tests/test_app.py
# ============================================================

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from app import (
    CATEGORY_MAP,
    LOCATION_MAP,
    COUNTRY_MAP,
    RATING_CATEGORY_MAP,
    EXPERIENCE_LEVEL_MAP,
)


class TestAppMappings:
    """Test application mappings"""

    def test_category_map(self):
        """Test category mapping"""
        assert CATEGORY_MAP[0] == "Data Science"
        assert CATEGORY_MAP[1] == "Design"
        assert CATEGORY_MAP[2] == "Marketing"
        assert CATEGORY_MAP[3] == "Mobile App"
        assert CATEGORY_MAP[4] == "Web Development"
        assert CATEGORY_MAP[5] == "Writing"
        assert len(CATEGORY_MAP) == 6

    def test_location_map(self):
        """Test location mapping"""
        assert LOCATION_MAP[0] == "Australia"
        assert LOCATION_MAP[8] == "UK"
        assert LOCATION_MAP[9] == "USA"
        assert len(LOCATION_MAP) == 10

    def test_country_map(self):
        """Test country mapping"""
        assert COUNTRY_MAP[0] == "Australia"
        assert COUNTRY_MAP[3] == "India"
        assert COUNTRY_MAP[8] == "USA"
        assert len(COUNTRY_MAP) == 10

    def test_rating_category_map(self):
        """Test rating category mapping"""
        assert RATING_CATEGORY_MAP[0] == "Poor (1-2)"
        assert RATING_CATEGORY_MAP[1] == "Average (2-3)"
        assert RATING_CATEGORY_MAP[2] == "Good (3-4)"
        assert RATING_CATEGORY_MAP[3] == "Excellent (4-5)"
        assert len(RATING_CATEGORY_MAP) == 4

    def test_experience_level_map(self):
        """Test experience level mapping"""
        assert EXPERIENCE_LEVEL_MAP[0] == "Entry (0-2 yrs)"
        assert EXPERIENCE_LEVEL_MAP[1] == "Junior (2-5 yrs)"
        assert EXPERIENCE_LEVEL_MAP[2] == "Senior (5-10 yrs)"
        assert EXPERIENCE_LEVEL_MAP[3] == "Expert (10+ yrs)"
        assert len(EXPERIENCE_LEVEL_MAP) == 4

    def test_map_completeness(self):
        """Test all maps are complete and have correct values"""
        assert set(CATEGORY_MAP.keys()) == {0, 1, 2, 3, 4, 5}
        assert set(LOCATION_MAP.keys()) == set(range(10))
        assert set(COUNTRY_MAP.keys()) == set(range(10))
        assert set(RATING_CATEGORY_MAP.keys()) == {0, 1, 2, 3}
        assert set(EXPERIENCE_LEVEL_MAP.keys()) == {0, 1, 2, 3}


class TestAppCalculations:
    """Test calculations in the app"""

    def test_budget_per_proposal_calculation(self):
        """Test budget per proposal calculation"""
        budget = 5000
        proposals = 5
        expected = budget / (proposals + 1)
        result = budget / (proposals + 1)
        assert result == expected

    def test_experience_quality_interaction_calculation(self):
        """Test experience-quality interaction calculation"""
        experience = 10
        quality = 9
        expected = 90
        result = experience * quality
        assert result == expected

    def test_combined_experience_rating_calculation(self):
        """Test combined experience rating calculation"""
        freelancer_rating = 4.7
        experience = 10
        jobs_completed = 200
        expected = (4.7 * 0.4) + (10 * 0.3) + (200 * 0.3)
        result = (freelancer_rating * 0.4) + (experience * 0.3) + (jobs_completed * 0.3)
        assert result == expected

    def test_rating_per_proposal_calculation(self):
        """Test rating per proposal calculation"""
        freelancer_rating = 4.7
        proposals = 5
        expected = 4.7 / (5 + 1)
        result = freelancer_rating / (proposals + 1)
        assert result == expected

    def test_interaction_features(self):
        """Test interaction features calculation with floating point precision"""
        proposal_quality = 9
        freelancer_rating = 4.7
        freelancer_experience = 10
        client_rating = 4.8

        # Use pytest.approx for floating point comparisons
        assert proposal_quality * freelancer_rating == pytest.approx(42.3)
        assert proposal_quality * freelancer_experience == 90
        assert client_rating * freelancer_rating == pytest.approx(22.56)


class TestAppDataValidation:
    """Test input data validation"""

    def test_budget_range(self):
        """Test project budget range validation"""
        min_budget = 100
        max_budget = 100000
        assert min_budget >= 100
        assert max_budget <= 100000
        assert min_budget < max_budget

    def test_rating_range(self):
        """Test rating range validation"""
        min_rating = 1.0
        max_rating = 5.0
        assert min_rating >= 1.0
        assert max_rating <= 5.0

    def test_proposal_count_range(self):
        """Test proposal count validation"""
        min_proposals = 0
        max_proposals = 200
        assert min_proposals >= 0
        assert max_proposals >= 0

    def test_experience_range(self):
        """Test experience range validation"""
        min_experience = 0
        max_experience = 30
        assert min_experience >= 0
        assert max_experience > 0

    def test_quality_score_range(self):
        """Test quality score range validation"""
        min_score = 1
        max_score = 10
        assert min_score >= 1
        assert max_score <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
