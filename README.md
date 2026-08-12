# DynaSID-POI

面向下一地点（Next-POI）推荐的动态时空语义 ID 方法。本仓库在真实的 Foursquare-NYC 与 Foursquare-TKY 数据上复现 GNPR-SID V2 的完整**语义 ID 阶段**，并在严格防止数据泄漏的统一协议下评估冻结参数的 DynaSID 检索模型。

需要特别说明：本项目**没有训练 GNPR-SID 的大语言模型生成器**。因此，文中的性能比较对象是相同实验协议下的静态语义 ID 检索基线，而不是 GNPR-SID 论文中的完整端到端生成模型。

## 本版本的主要改进

- 在 NYC 和 TKY 上完成 GNPR-SID V2 CRQ-VAE 训练，使用三个大小均为 64 的残差码本。
- 通过 `--city-prefix` 支持不同城市的数据加载。
- 完成冻结参数的跨城市验证：在 NYC 上确定 DynaSID 权重，再直接应用于 TKY，测试集不参与调参。
- 提供 8 组消融实验，覆盖语义 ID 历史、用户重访、近期记忆、POI 转移、多尺度局部性、空间机制、安全校准及停留/离开控制。
- 提供用户级配对 Bootstrap 置信区间、语义 ID 碰撞分析、逐案例改进/退化分析及 CPU 计时审计。
- 生成白色背景、图内无顶部标题的报告用图。
- 增加经过校验的 JSON 配置、数据泄漏检查、一键 CPU 实验、图表再生成及自动测试。

## 真实数据主要结果

| 城市 / 方法 | Acc@1 | Recall@10 | NDCG@10 | 不可达率@10 | 平均距离@10 |
|---|---:|---:|---:|---:|---:|
| NYC / 静态 GNPR-SID 检索 | 0.1746 | 0.4038 | 0.2860 | 0.3800 | 9.02 km |
| **NYC / DynaSID-v8** | **0.1925** | **0.5324** | **0.3613** | **0.0031** | **0.92 km** |
| TKY / 静态 GNPR-SID 检索 | 0.0955 | 0.2556 | 0.1720 | 0.3904 | 9.12 km |
| **TKY / DynaSID-v8（冻结参数）** | **0.1427** | **0.4616** | **0.2959** | **0.0011** | **0.68 km** |

在未参与调参的 TKY 确认集上，用户级配对差值为：Acc@1 提升 0.0472（95% CI [0.0355, 0.0591]），Recall@10 提升 0.2061（[0.1865, 0.2258]），NDCG@10 提升 0.1239（[0.1122, 0.1350]），不可达率@10 降低 0.3893（[-0.3992, -0.3793]），平均距离@10 降低 8.44 km（[-8.60, -8.28]）。

## 语义 ID 审计

| 城市 | POI 数 | 唯一语义 ID 数 | 碰撞率 | L1/L2/L3 已用编码数 | L1/L2/L3 困惑度 |
|---|---:|---:|---:|---:|---:|
| NYC | 4,980 | 4,513 | 9.38% | 64 / 64 / 64 | 62.36 / 62.83 / 63.06 |
| TKY | 7,793 | 6,022 | 22.73% | 52 / 64 / 64 | 20.26 / 55.72 / 58.89 |

TKY 的第一层码本分布不均衡，且语义 ID 碰撞率更高。这为“使用动态用户状态与空间上下文对静态语义 ID 进行条件化修正”提供了直接的跨城市证据。

## 仓库结构

```text
src/
  dynasid/                       可复用的配置、地理计算、指标与数据校验工具
  full_sid_v2_experiment.py     训练并审计 GNPR-SID V2 CRQ-VAE
  poi_features.py               仅使用训练集构建 POI 特征
  sid_downstream_experiment.py  严格协议下的基线与 DynaSID 实验
  extended_evaluation.py        跨城市消融、Bootstrap、案例及效率评估
checkpoints/
  crqvae_best.pt                NYC 最优检查点
results/
  sid_quality/                  NYC 语义 ID 与码本审计结果
  tky_sid/                      TKY 语义 ID 与码本审计结果
  downstream/                   NYC 严格协议原始结果
  cross_city/NYC/               NYC 扩展审计与消融实验
  cross_city/TKY/               TKY 冻结参数确认实验
figures/                        用于报告的可视化图
configs/                        NYC 与冻结参数 TKY 配置
scripts/                        数据校验、CPU 流程及图表生成脚本
tests/                          配置、距离、指标、Bootstrap 与泄漏测试
docs/                           数据说明、模型说明、局限性与复现指南
```

## 数据准备

使用经 GETNext/LLM4POI 兼容流程预处理的 Foursquare 数据划分。本仓库不重新分发原始签到数据。每个城市目录应包含 `{PREFIX}_train.csv`、`{PREFIX}_val.csv` 和 `{PREFIX}_test.csv`，其中前缀通常为 `NYC` 或 `TKY`。

必要字段包括 `user_id`、`POI_id`、`POI_catid_code`、`latitude`、`longitude`、`trajectory_id` 和 `local_time`。

## 环境与复现

```bash
python -m venv .venv
# Windows：.venv\Scripts\activate
# Linux/macOS：source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/wds1996/GNPR-SID.git ../GNPR-SID
```

安装本项目并运行 CPU 自动测试：

```bash
pip install -e .
python -m unittest discover -s tests -v
```

训练前校验城市数据：

```bash
python scripts/validate_dataset.py \
  --data-dir /path/to/city \
  --city-prefix NYC \
  --out artifacts/nyc_dataset_audit.json
```

一键 CPU 流程会记录执行命令、耗时和运行环境：

```bash
python scripts/run_cpu_pipeline.py \
  --config configs/nyc.json \
  --data-dir /path/to/city \
  --gnpr-root ../GNPR-SID \
  --work-dir artifacts/nyc_run
```

训练单个城市的完整语义 ID 阶段：

```bash
python src/full_sid_v2_experiment.py \
  --data-dir /path/to/city \
  --city-prefix TKY \
  --gnpr-root ../GNPR-SID \
  --out artifacts/tky_sid \
  --epochs 100 \
  --batch-size 256
```

运行冻结参数的扩展评估：

```bash
python src/extended_evaluation.py \
  --data-dir /path/to/city \
  --city-prefix TKY \
  --sid-dir artifacts/tky_sid \
  --out artifacts/extended_tky
```

运行严格协议下的原始基线：

```bash
python src/sid_downstream_experiment.py \
  --data-dir /path/to/city \
  --city-prefix NYC \
  --sid-dir artifacts/nyc_sid \
  --out artifacts/baselines \
  --baselines-only
```

## 评估协议

- POI 特征、用户历史与转移统计均仅使用各城市训练集拟合。
- 模型选择和安全校准只使用验证集。
- TKY 确认实验冻结在 NYC 上确定的 DynaSID-v8 权重。
- 指标先按用户宏平均；置信区间通过 5,000 次用户级重采样计算。
- 当推荐 POI 与当前位置的 Haversine 距离超过任务设定的 10 km 阈值时，记为不可达推荐。
- 通过验证集选择的停留/离开机制，允许连续访问同一个 POI。
- CPU 计时不包含所有模型共享的特征构建，只应理解为数量级层面的效率审计。

## 研究边界与局限性

本项目完成了完整的语义 ID 阶段和下游检索实验，但不是完整的 LLM 生成式复现。10 km 可达性阈值是任务定义的代理指标，不能替代真实路程时间。TKY 的最优码本轮次及第一层码本集中现象说明，未来仍需研究城市条件化量化。GNPR-SID 官方仓库目前没有明确的软件许可证，因此本仓库从单独克隆的官方代码中加载 CRQ-VAE，而不直接重新分发其源码。

使用本项目材料时，请同时引用 Dongsheng Wang 等人的论文《Generative Next POI Recommendation with Semantic ID》（KDD 2025）及本仓库。

完整 CPU 流程见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)，结论适用范围见 [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)。
