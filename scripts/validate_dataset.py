"""无需训练模型，直接校验单个城市的原始数据划分。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynasid.validation import audit_city, write_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--city-prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_city(args.data_dir, args.city_prefix)
    write_audit(audit, args.out)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
