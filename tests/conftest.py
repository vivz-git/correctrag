"""
Pytest configuration for CorrectRAG test suite.
"""

from pathlib import Path
import sys

# Ensure backend directory is in sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
