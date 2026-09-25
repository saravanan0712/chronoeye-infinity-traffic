"""
Runner to execute focused ANPR pytest suite programmatically.
"""
import os, sys, pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

if __name__ == "__main__":
    test_files = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "tests", "test_stage4_anpr.py")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "tests", "test_alpr_pipeline.py")),
    ]
    print(f"Running pytest on: {test_files}")
    exit_code = pytest.main(["-v"] + test_files)
    sys.exit(exit_code)
