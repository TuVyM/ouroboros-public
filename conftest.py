"""
pytest conftest — adds the repo root to sys.path so tests can import
project modules without manual sys.path.insert() in each test file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
