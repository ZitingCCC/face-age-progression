"""Run a single SAM smoke test with explicit external assets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.smoke_cli import main

if __name__ == '__main__':
    main('sam')
