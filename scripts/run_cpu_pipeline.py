"""用于数据校验、语义 ID 训练和评估的一键 CPU 流程。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from dynasid.config import load_config
from dynasid.validation import audit_city, write_audit


def run(command: list[str], log_path: Path) -> dict:
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n\n")
        stream.flush()
        completed = subprocess.run(command, text=True, stdout=stream, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"命令执行失败（返回码 {completed.returncode}），详见 {log_path}")
    return {"command": command, "seconds": elapsed, "log": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gnpr-root", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sid-dir", type=Path, help="复用已有的语义 ID 目录")
    parser.add_argument("--skip-sid-training", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    sid_dir = (args.sid_dir or work / "sid").resolve()
    audit = audit_city(args.data_dir, config.city_prefix)
    write_audit(audit, work / "dataset_audit.json")

    steps = []
    if not args.skip_sid_training:
        if args.gnpr_root is None:
            raise ValueError("未使用 --skip-sid-training 时必须提供 --gnpr-root")
        steps.append(run([
            sys.executable, str(root / "src" / "full_sid_v2_experiment.py"),
            "--data-dir", str(args.data_dir), "--city-prefix", config.city_prefix,
            "--gnpr-root", str(args.gnpr_root), "--out", str(sid_dir),
            "--epochs", str(config.epochs), "--batch-size", str(config.batch_size),
            "--patience", str(config.patience), "--seed", str(config.seed),
        ], work / "logs" / "01_sid_training.log"))
    elif not (sid_dir / "static_semantic_ids.csv").exists():
        raise FileNotFoundError(f"未找到已有语义 ID：{sid_dir / 'static_semantic_ids.csv'}")

    steps.append(run([
        sys.executable, str(root / "src" / "sid_downstream_experiment.py"),
        "--data-dir", str(args.data_dir), "--city-prefix", config.city_prefix,
        "--sid-dir", str(sid_dir), "--out", str(work / "baselines"), "--baselines-only",
    ], work / "logs" / "02_baselines.log"))
    steps.append(run([
        sys.executable, str(root / "src" / "extended_evaluation.py"),
        "--data-dir", str(args.data_dir), "--city-prefix", config.city_prefix,
        "--sid-dir", str(sid_dir), "--out", str(work / "extended"),
    ], work / "logs" / "03_extended_evaluation.log"))

    manifest = {
        "config": str(args.config.resolve()), "data_dir": str(args.data_dir.resolve()),
        "sid_dir": str(sid_dir), "city": config.city_prefix, "steps": steps,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "executable": sys.executable, "cpu_count": os.cpu_count(),
        },
    }
    (work / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
