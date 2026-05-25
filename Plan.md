# 口内照三维重建与牙周炎诊断项目开发计划（Python）

## 1. 项目目标
基于 [Goal.md](Goal.md) 的定义，使用二维标准口内照（多视角）+ 语义区域约束（牙齿/牙龈），构建高精度三维重建流程，并在三维空间下进行牙周炎诊断分析。

关键参考：
- 标注与分割：SAM2.1 + X-AnyLabeling
- 三维重建：VGGT
- 诊断策略：结合 [文献.md](文献.md) 中 HC-Net / HC-Net+ 的“局部-全局-融合”思想

## 2. 当前数据现状（初步扫描）
- 数据位置：`dataset/`
- 初步统计：
  - 总文件数约 1828
  - JPG 图像约 1617
  - PPT 约 30
  - Mac 元数据文件（`._*`、`.DS_Store`）约 331
- 数据结构特征：`dataset/大类/病例编号/多张JPG(+ppt)`，适合组织成“病例级多视角序列”。

## 3. 总体技术路线（Python）
1. 数据治理与标准化
2. 牙齿/牙龈语义标注与自动分割
3. 多视角三维重建（VGGT）
4. 语义约束融合（将2D掩膜映射到3D）
5. 三维特征提取与牙周炎诊断
6. 评估、可视化与复现实验

## 4. 分阶段开发计划

### 阶段 A：环境与工程初始化（第 1 周）- 已完成
目标：建立可复现工程框架，确保后续训练/推理可跑通。

任务：
- 创建 Python 项目结构：
  - `src/data/`（数据清洗与索引）
  - `src/seg/`（分割推理/训练）
  - `src/recon/`（VGGT 接口与重建）
  - `src/diag/`（诊断模型与融合策略）
  - `src/eval/`（指标评估）
  - `configs/`（YAML 配置）
  - `scripts/`（一键运行脚本）
  - `outputs/`（结果与日志）
- 固定依赖版本（PyTorch、OpenCV、numpy、pandas、scikit-learn、open3d、plotly 等）。
- 建立实验追踪方案（建议 MLflow 或 Weights & Biases 二选一）。

交付：
- `requirements.txt` + `README.md` + 最小可运行脚本。

当前进展（2026-05-05）：
- 已建立 `src/data/`、`src/seg/`、`src/recon/`、`src/diag/`、`src/eval/`、`configs/`、`scripts/`、`outputs/` 等工程骨架。
- 已固定首版依赖于 `requirements.txt`，包含 PyTorch、OpenCV、numpy、pandas、scikit-learn、open3d、plotly、MLflow 等。
- 已提供 `scripts/run_data_pipeline.ps1` 作为阶段 B 最小可运行数据流水线入口。
- 已在 `configs/default.yaml` 中记录默认路径、清洗规则、划分比例与 MLflow 追踪后端。

### 阶段 B：数据治理与质控（第 1-2 周）- 已完成
目标：把原始数据变成可训练、可追踪的数据资产。

任务：
- 编写 `src/data/scan_dataset.py`：统计每个病例图像数、分辨率、缺失情况。
- 编写 `src/data/clean_dataset.py`：
  - 过滤 `._*`、`.DS_Store` 等噪声文件
  - 识别并记录损坏图片
  - 建立统一命名规则（病例ID、视角ID、拍摄序号）
- 从 `SIOP quality evaluation.xlsx` 读取质量分级，并生成 `metadata.csv`。
- 形成病例级划分（train/val/test，按患者分层，避免泄漏）。

交付：
- `data/processed/`（清洗后数据）
- `data/splits/*.csv`（划分清单）
- `outputs/reports/data_quality_report.md`

当前进展（2026-05-05）：
- 已实现 `src/data/scan_dataset.py`，输出 `outputs/reports/dataset_scan.json`、`case_image_summary.csv` 与报告基础统计。
- 已实现 `src/data/build_metadata.py`，从 `dataset/SIOP quality evaluation.xlsx` 读取 IQS 与 OE sheet，生成 `data/processed/metadata.csv`，包含病例级 IQS 评分、overall evaluation、gingival index 与质量桶。
- 已增强 `src/data/clean_dataset.py`，过滤 Mac 元数据和非图像文件，验证图片完整性，按病例与视角生成统一命名，并输出 `data/processed/clean_manifest.csv`。
- 已增强 `src/data/build_splits.py`，按病例级划分 train/val/test，并可根据 `gingival_index` 做分层，避免同一病例跨集合泄漏。
- 已实现 `src/data/generate_data_report.py`，汇总扫描、清洗、metadata 与 split 结果到 `outputs/reports/data_quality_report.md`。
- 已运行完整流水线，当前统计：总文件 1828，图像文件 1340，有效图像 1339，异常图像 1，PPT 150，Mac 元数据 331，病例 150。
- 当前划分：train 104 例/928 张图，val 20 例/177 张图，test 26 例/234 张图；按 `gingival_index` 分层成功，跨 split 病例泄漏检查为 0。
- 质控发现：`dataset/4/4-10/4-10-8.jpg` 触发 Pillow `DecompressionBombError`，已作为异常图像记录而非复制进 processed 数据。

### 阶段 C：牙齿/牙龈标注与分割（第 2-4 周）
目标：获得稳定的牙齿/牙龈语义掩膜，为三维约束提供输入。

任务：
- 使用 X-AnyLabeling + SAM2.1 生成初始标注（牙齿、牙龈）。
- 抽样人工校正并形成高质量金标准子集。
- 训练/微调分割模型（可先从 SAM2.1 推理结果蒸馏到轻量模型）。
- 输出每张图对应 mask，并记录置信度。

指标：
- Dice（tooth/gingiva）
- IoU（tooth/gingiva）

交付：
- `outputs/seg/masks/`
- `outputs/seg/metrics.json`

当前进展（2026-05-25）：
- 已定义 tooth/gingiva/background 的单通道 PNG 掩膜格式，并在 `configs/default.yaml` 中记录阶段 C 默认路径、类别 ID 与 baseline backend。
- 已实现 `src/seg/build_annotation_queue.py`，可从 `data/splits/*.csv` 按 split 与 `gingival_index` 分层抽样，生成 `outputs/seg/annotation_queue.csv` 与 `outputs/seg/annotation_schema.json`，作为 X-AnyLabeling/SAM2.1 人工校正闭环的输入清单。
- 已实现 `src/seg/baseline_segment.py`，提供可替换的 Pillow/NumPy 颜色阈值 baseline，批量输出 tooth/gingiva 语义 mask 到 `outputs/seg/masks/`，并在 `outputs/seg/pred_manifest.csv` 记录每张图的 mask 路径、coverage 与 confidence。
- 已实现 `src/seg/evaluate_masks.py`，在存在人工金标准 mask 时计算 tooth/gingiva Dice 与 IoU；当前无金标准时仍会生成 `outputs/seg/metrics.json`，记录预测数量、coverage、confidence 与待补标注状态。
- 已新增 `scripts/run_seg_pipeline.ps1`，阶段 C 可通过单命令跑通。当前实现是工程 baseline，不替代最终 SAM2.1/轻量模型精度目标；后续模型只需复用 `pred_manifest.csv` 与 mask 目录契约。

### 阶段 D：多视角三维重建（第 4-6 周）
目标：基于病例多视角图像完成可用的3D重建。

任务：
- 打通 VGGT 推理流程，封装 `src/recon/run_vggt.py`。
- 建立病例级输入打包逻辑（按视角顺序与质量筛选）。
- 输出点云/网格，并保存相机参数（若可导出）。
- 记录重建失败模式（反光、遮挡、模糊、视角缺失）。

指标：
- 重建成功率
- 几何完整度（可见区域覆盖率）
- 重建一致性（多次推理稳定性）

交付：
- `outputs/recon/{case_id}/`（点云、网格、日志）

### 阶段 E：语义约束融合到三维（第 6-8 周）
目标：让3D模型具备“牙齿/牙龈”语义，支撑病灶定位和诊断。

任务：
- 将2D mask 按相机几何投影/反投影到3D点云。
- 解决多视角语义冲突（投票/置信度加权/时序优先）。
- 生成语义点云（每点包含类别与置信度）。
- 对牙齿与牙龈区域提取几何和纹理特征。

指标：
- 语义一致性分数（跨视角）
- 关键区域覆盖率（牙龈边缘、牙周高风险区域）

交付：
- `outputs/recon_semantic/{case_id}/`

### 阶段 F：三维牙周炎诊断建模（第 8-10 周）
目标：借鉴文献的局部-全局融合思想，实现病例级诊断。

任务：
- 局部分支：牙齿级/牙龈级区域特征建模。
- 全局分支：病例级3D整体表征（形态+纹理）。
- 融合层：采用 Noisy-OR 或门控融合，将局部风险聚合为病例风险。
- 输出分期/风险评分（按你现有标签体系）。

指标：
- AUROC、AUPRC、Sensitivity、Specificity
- 与2D-only基线对比增益

交付：
- `outputs/diag/`（模型权重、预测结果、ROC曲线）

### 阶段 G：评估、解释与论文化结果（第 10-11 周）
目标：形成可汇报、可复现实验结论。

任务：
- 进行消融实验：
  - 无语义约束 vs 有语义约束
  - 仅2D vs 2D+3D
  - 不同融合策略对比
- 生成可视化：3D语义叠加、病例级风险热力图。
- 固化实验脚本（单命令复现实验）。

交付：
- `outputs/final_report.md`
- 可复现实验命令清单

## 5. 里程碑与验收标准
- M1（第2周）：完成数据清洗与划分，形成可训练数据集。
- M2（第4周）：分割模型达到可用精度（Dice/IoU 达到基线）。
- M3（第6周）：VGGT 在多数病例可稳定重建。
- M4（第8周）：完成2D语义到3D融合并可视化。
- M5（第10周）：完成诊断模型与基线对比。
- M6（第11周）：完成消融与最终报告。

## 6. 风险与应对
- 视角不标准/图像质量波动：引入质量筛选与加权。
- 反光和遮挡导致重建失败：增加数据增强与失败重试策略。
- 标注成本高：采用“自动预标注 + 人工校正”闭环。
- 标签体系不足：先做二分类风险评分，再逐步扩展分期。

## 7. 近期执行清单（下一步）
1. 阶段 A/B 已完成，保留当前数据流水线作为后续训练与实验入口。
2. 阶段 C 工程闭环已完成：标注队列、类别 schema、baseline mask 推理、置信度记录与 metrics 输出均已具备。
3. 下一步优先抽取 `outputs/seg/annotation_queue.csv` 中的样本，用 X-AnyLabeling + SAM2.1 进行人工校正，并把二值金标准 mask 放到清单指定路径。
4. 用人工金标准复跑 `src.seg.evaluate_masks`，得到真实 tooth/gingiva Dice 与 IoU；再决定是否接入 SAM2.1 批量推理或训练轻量蒸馏模型。
5. 复核 `dataset/4/4-10/4-10-8.jpg` 是否为真实可用超大图；若需要保留，可后续加入受控降采样策略。

---
本计划为“先可跑通、再提精度”的工程路径。后续将按每个里程碑更新计划与实际进展偏差。
