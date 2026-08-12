import unittest
from pathlib import Path
import pandas as pd

from dynasid.validation import audit_city


def frame(trajectory):
    return pd.DataFrame({
        "user_id": [1], "POI_id": [10], "POI_catid_code": [3], "latitude": [40.7],
        "longitude": [-74.0], "trajectory_id": [trajectory],
        "local_time": [f"2020-01-0{trajectory}T10:00:00Z"],
    })


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / "_validation_data"
        self.root.mkdir(exist_ok=True)

    def tearDown(self):
        for path in self.root.glob("*.csv"):
            path.unlink()
        self.root.rmdir()

    def test_split_disjoint_audit(self):
        for split, trajectory in zip(("train", "val", "test"), (1, 2, 3)):
            frame(trajectory).to_csv(self.root / f"NYC_{split}.csv", index=False)
        result = audit_city(self.root, "NYC")
        self.assertEqual(result["train_test_trajectory_overlap"], 0)

    def test_duplicate_event_leakage_fails(self):
        frame(1).to_csv(self.root / "NYC_train.csv", index=False)
        frame(2).to_csv(self.root / "NYC_val.csv", index=False)
        frame(1).to_csv(self.root / "NYC_test.csv", index=False)
        with self.assertRaises(ValueError):
            audit_city(self.root, "NYC")


if __name__ == "__main__":
    unittest.main()
