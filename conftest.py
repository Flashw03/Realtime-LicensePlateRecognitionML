"""Make the project root importable so tests can `import src...`."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
