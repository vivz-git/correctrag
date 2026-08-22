"""
Pytest configuration for CorrectRAG test suite.
"""

from pathlib import Path
import sys

# Ensure root and backend directories are in sys.path
root_path = Path(__file__).resolve().parent.parent
backend_path = root_path / "backend"
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
