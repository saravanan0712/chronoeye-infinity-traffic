import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["backend/tests/test_stage4_anpr.py", "-v", "--tb=short"]))
