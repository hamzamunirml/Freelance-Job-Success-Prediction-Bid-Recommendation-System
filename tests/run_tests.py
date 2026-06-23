# ============================================================
# Test Runner Script
# File: tests/run_tests.py
# ============================================================

import subprocess
import sys
import os


def run_all_tests():
    """Run all tests with coverage"""
    print("🧪 Running all tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode


def run_unit_tests():
    """Run only unit tests"""
    print("🧪 Running unit tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-m", "unit", "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    return result.returncode


def run_integration_tests():
    """Run only integration tests"""
    print("🧪 Running integration tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-m", "integration", "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    return result.returncode


def run_with_coverage():
    """Run tests with coverage report"""
    print("🧪 Running tests with coverage...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    return result.returncode


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 FREELANCE JOB SUCCESS PREDICTION - TEST SUITE")
    print("=" * 60)

    print("\n1️⃣ Running Unit Tests...")
    run_unit_tests()

    print("\n2️⃣ Running Integration Tests...")
    run_integration_tests()

    print("\n3️⃣ Running All Tests with Coverage...")
    run_with_coverage()

    print("\n✅ All tests completed!")
