"""
ChronoEye Infinity - Command-Line Entry Point for Research Benchmark Execution
Executes: python -m backend.app.research.benchmark
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.research.runner import ResearchBenchmarkRunner
from app.research.serializer import serialize_benchmark_results
from app.research.report_generator import generate_research_markdown_report


def main():
    print("Starting ChronoEye Infinity Empirical Research Evaluation Benchmark...")
    runner = ResearchBenchmarkRunner(seeds=[42, 101, 202, 303, 404], steps=20)
    results = runner.run_all()
    serialize_benchmark_results(results, output_dir="e:/chronoeye/data/research")
    generate_research_markdown_report(results, output_path="e:/chronoeye/docs/RESEARCH_BENCHMARK_REPORT.md")
    print("Empirical Research Evaluation Benchmark complete!")
    print("Data serialized to e:/chronoeye/data/research/ and report generated at e:/chronoeye/docs/RESEARCH_BENCHMARK_REPORT.md")


if __name__ == "__main__":
    main()
