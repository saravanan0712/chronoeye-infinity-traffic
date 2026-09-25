import unittest

loader = unittest.TestLoader()
suite = loader.discover("backend/tests")
print(f"Total tests discovered by unittest: {suite.countTestCases()}")
