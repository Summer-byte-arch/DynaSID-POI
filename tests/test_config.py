import json
import unittest
from pathlib import Path

from dynasid.config import load_config


class ConfigTests(unittest.TestCase):
    def test_released_nyc_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "nyc.json")
        self.assertEqual(config.city_prefix, "NYC")
        self.assertEqual(config.codebook_sizes, (64, 64, 64))
        self.assertEqual(len(config.frozen_weights), 17)

    def test_invalid_radius_order(self):
        root = Path(__file__).resolve().parents[1]
        raw = json.loads((root / "configs" / "nyc.json").read_text(encoding="utf-8"))
        raw["spatial_radii_km"] = [10, 2, 5]
        path = Path(__file__).resolve().parent / "_bad_config.json"
        try:
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
