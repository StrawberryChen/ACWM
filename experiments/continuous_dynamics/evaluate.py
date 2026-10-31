"""Convenience evaluation entry point; same strict checkpoint/config contract."""
import sys
from .train import main

if __name__ == "__main__":
    sys.argv.extend(["--mode", "evaluate"])
    main()
