import yaml
from pathlib import Path

with open(Path(__file__).parent / "params.yaml", "r") as fp:
    params = yaml.safe_load(fp)
