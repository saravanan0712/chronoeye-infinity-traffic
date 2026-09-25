"""
Backward-compatibility import package for app.benchmarks forwarding to benchmarks package.
"""

import sys
import benchmarks

# Expose submodules seamlessly under app.benchmarks
for attr in dir(benchmarks):
    if not attr.startswith("__"):
        globals()[attr] = getattr(benchmarks, attr)
