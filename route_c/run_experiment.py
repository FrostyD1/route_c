#!/usr/bin/env python3
"""
Route C Main Entry Point
========================
Run this script from the route_c directory:
    python run_experiment.py

Or from parent directory:
    python -m route_c.run_experiment
"""

import sys
import os

# Add route_c to path for direct imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now run the unified experiment
from experiments.exp_unified import main

if __name__ == "__main__":
    main()
