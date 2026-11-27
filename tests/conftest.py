import sys
from pathlib import Path

# Make the controller modules (config, logic, sentinels, env) importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controllers" / "sentinel_agent"))
