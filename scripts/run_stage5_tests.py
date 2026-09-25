"""
Runner to execute Stage 5 Re-ID and Journey test suite via pytest.
"""
import os, sys, pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

if __name__ == "__main__":
    test_files = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "tests", "test_stage5_reid_journey.py")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "tests", "test_reid_pipeline.py")),
    ]
    print(f"Executing pytest on Stage 5 tests: {test_files}")
    exit_code = pytest.main(["-v"] + test_files)
    sys.exit(exit_code)
