# CPU 实验复现指南

## 配置运行环境

```bash
python -m venv .venv
# Windows：.venv\Scripts\activate
# Linux/macOS：source .venv/bin/activate
pip install -e .
git clone https://github.com/wds1996/GNPR-SID.git ../GNPR-SID
```

## 校验数据

```bash
python scripts/validate_dataset.py --data-dir /path/to/NYC --city-prefix NYC --out artifacts/nyc_dataset_audit.json
```

## 完整 CPU 流程

```bash
python scripts/run_cpu_pipeline.py --config configs/nyc.json --data-dir /path/to/NYC --gnpr-root ../GNPR-SID --work-dir artifacts/nyc_run
```

如需复用已有语义 ID：

```bash
python scripts/run_cpu_pipeline.py --config configs/tky_frozen.json --data-dir /path/to/TKY --sid-dir artifacts/tky_sid --work-dir artifacts/tky_confirmation --skip-sid-training
```

每次运行都会保存数据审计、命令日志、包含环境信息的运行清单、基线结果和扩展评估结果。

## 重新生成图表并运行测试

```bash
python scripts/generate_figures.py --results-dir results --out artifacts/figures
python -m unittest discover -s tests -v
```
