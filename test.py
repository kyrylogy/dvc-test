from train import load_model
from dataset import get_dataset
import json
from pathlib import Path

output_dir = Path("./data/test")
output_dir.mkdir(exist_ok=True, parents=True)

def test():
    model = load_model()
    _, _, test_ds = get_dataset()

    metrics = model.evaluate(test_ds, return_dict=True)
    with open(output_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp)

if __name__ == "__main__":
    test()    