# MoE PLMs: S669 prediction and Colab usage instructions

This directory provides the inference entry for MoE PLMs on the S669 protein stability change (Δ Δ G) dataset. The model integrates pre calculated features from ProstT5, ESMC, and ESM2, loads existing checkpoints, and outputs sample by sample prediction and regression evaluation metrics.

S669 inference will restore the network parameters, expert routing methods, and MinMaxScaler during training, without retraining or fitting the scaler on the test set. The input package already contains weights and test features, there is no need to re extract embeddings for this example.

> **Current version description: * * The notebook currently has the local VS Code/Jupyter path enabled, and the path needs to be modified before use; The Colab upload and Drive code are currently being commented out. In addition, the hash of the weights in the attached ZIP is inconsistent with the list, and the list needs to be checked and updated according to the following text before command-line inference. The current notebook cannot be directly treated as a one click version of Colab that does not require configuration.

## 目录结构

```text
MoE_colab/
├── README.md
├── S669_Colab_Predict.ipynb       # 环境检查、数据准备、模型定义、推理和结果打包
├── predict.py                    # 独立命令行推理入口
├── S669_inputs.zip               # S669 权重、元数据、三路特征与哈希清单
├── S669_local_inputs/            # 已解压的输入及历史运行结果
│   ├── weights/final_model.pt
│   ├── data/
│   │   ├── S669.csv
│   │   ├── S669_ProstT5.pt
│   │   ├── S669_ESMC.pt
│   │   └── S669_ESM2.pkl
│   ├── input_manifest.json
│   └── outputs/
└── S571_inputs/                  # S571（ΔTm）迁移学习相关文件，见末节说明
    ├── DTm-transfer.py
    ├── best_model.pth
    ├── S571.csv
    ├── S571_protT5.pt
    ├── S571_ESMC_600M.pt
    └── S571_ESM2.pkl
```

`S669_inputs.zip` 内直接包含 `weights/`、`data/` 和 `input_manifest.json`，没有额外的顶层文件夹。`S669_local_inputs/` 可从该 ZIP 重新解压生成。

## 输入与模型

| 输入 | 文件 | 特征维度 |
| --- | --- | --- |
| 样本元数据 | `data/S669.csv` | 669 条记录，包含 `ID` 和 `target` |
| ProstT5 特征 | `data/S669_ProstT5.pt` | 1024 |
| ESMC 特征 | `data/S669_ESMC.pt` | 1152 |
| ESM2 特征 | `data/S669_ESM2.pkl` | 1280 |
| 完整模型检查点 | `weights/final_model.pt` | 网络权重、配置及训练集 scaler |

三路特征先投影到统一维度，经 MLP 交互和稀疏 MoE 后融合，再由回归头输出预测值。程序支持 `mlp_moe_attention` 和 `mlp_moe_reg_head` 两种结构，按检查点配置恢复模型及路由方式。

特征按 `ID` 与元数据对齐，输出保留元数据的行顺序和重复测量记录。特征文件中的 ID 必须唯一，且覆盖元数据中的所有 ID；特征列和维度必须与检查点一致。

## 本地命令行运行

以下命令均在 **`MoE_colab` 目录**执行。

### 1. 安装依赖

```bash
pip install environment.yml
```

To use a notebook, it is necessary to prepare a Jupyter or VS Code notebook runtime environment.

If it has been confirmed that the input package provided with this directory is being used, the following code can be executed in Python or notebook. It will first verify the weight hash and all data files, and then backup and update the weight list records. **Hash updates only make the list consistent with the file, without proving the source of weights or the correctness of the training process. **For weights from other sources or new training, the complete input package and list exported together should be used.

```python
from pathlib import Path
import hashlib
import json
import shutil

root = Path("S669_local_inputs")
manifest_path = root / "input_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
weight = root / "weights/final_model.pt"
digest = hashlib.sha256(weight.read_bytes()).hexdigest()
expected = "d65d0fe56f0396ccde3016f435c12918b4b0d3176517a4cd7a39bd9219c8d5f5"
if digest != expected:
    raise ValueError("权重与本文核对的随附版本不同，请检查来源和配套清单。")
for record in manifest["files"]:
    path = root / record["file"]
    if record["file"] != "weights/final_model.pt":
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"数据文件校验失败：{path}")
backup = manifest_path.with_suffix(".json.bak")
if not backup.exists():
    shutil.copy2(manifest_path, backup)
record = next(r for r in manifest["files"] if r["file"] == "weights/final_model.pt")
record.update(sha256=digest, bytes=weight.stat().st_size)
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
```

### 4. 执行预测

```bash
python predict.py --data-root S669_local_inputs
```

指定 CPU、批大小及输出目录：

```bash
python predict.py --data-root S669_local_inputs --device cpu --batch-size 32 --output-dir S669_local_inputs/outputs
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-root` | `predict.py` 所在目录 | 解压后的输入根目录；使用本目录结构时需显式传入 `S669_local_inputs` |
| `--output-dir` | `<data-root>/outputs` | 结果保存目录 |
| `--device` | `auto` | 自动选择 CUDA 或 CPU，也可指定 `cpu`、`cuda` |
| `--batch-size` | 检查点中的值 | 正整数；较小批次可降低推理显存占用 |

## 使用 notebook

打开 [S669_Colab_Predict.ipynb](S669_Colab_Predict.ipynb)，按以下方式配置输入，再按单元格顺序运行。

### 本地 Jupyter / VS Code

在“2. 上传权重与测试数据”下的代码单元格中，将 `zip_path` 从原作者的 Linux 绝对路径改为自己的路径。例如，在本目录启动 notebook 时：

```python
zip_path = Path("S669_inputs.zip").resolve()
```

随后运行环境检查、解压、模型定义、清单更新、预测和结果打包单元格。模型定义后有一个未编号的权重检查及清单更新单元格，它会加载完整检查点、备份清单，并更新权重哈希；请先确认输入包来源。最后一个单元格会生成 `S669_local_inputs/S669_prediction_results.zip`。

### Google Colab：上传 ZIP

1. 将 notebook 上传到 Colab 并打开。
2. 在数据准备单元格中，启用上半部分被注释的 Colab 代码，移除或注释下半部分本地 `# vscode` 代码，避免覆盖 `DATA_ROOT`。
3. 设置 `INPUT_MODE = 'upload'`、`UPLOAD_NEW_INPUTS = True`，运行时上传 `S669_inputs.zip`。该模式会将输入解压到 `/content/Colab_S669_Predict`。
4. 依次运行模型定义、权重检查及清单更新、预测单元格。`DEVICE = 'auto'` 会根据环境选择设备。
5. 如需将结果下载到电脑，用下面代码替换最后的结果打包单元格：

```python
import shutil
from google.colab import files

result_zip = shutil.make_archive(
    str(DATA_ROOT / "S669_prediction_results"),
    "zip",
    root_dir=output_dir,
)
files.download(result_zip)
```

### Google Colab：Google Drive

将 ZIP 解压后的完整目录放入 Drive。在启用的 Colab 数据准备代码中，设置：

```python
INPUT_MODE = 'drive'
DRIVE_DATA_ROOT = '/content/drive/MyDrive/MoE_colab/S669_local_inputs'
```

将示例路径替换为实际目录，挂载 Drive 后继续运行。该目录下须有 `weights/`、`data/` 和 `input_manifest.json`。默认输出也会写入这个目录下的 `outputs/`。

## 输出文件

| 文件 | 内容与生成条件 |
| --- | --- |
| `predictions.csv` | 每行一个预测，始终包含 `ID` 和 `prediction` |
| `metrics.csv` | 元数据有 `target` 列时生成整体回归指标 |
| `per_protein_metrics.csv` | 有 `target`，且所有 ID 均可按程序规则解析蛋白标识时生成分蛋白指标 |
| `run_info.json` | 输入文件及权重 SHA-256、模型配置、设备、批大小、部分软件版本和推理标记 |

当所有 ID 均匹配 `rcsb_XXXX_...` 或 `uniprot_XXXX_...` 形式时，预测表会增加 `protein_id`。有 `target` 时，还包含真实值、`residual = prediction - target` 和 `absolute_error`。

评估指标包括 MAE、RMSE、R²、Pearson、Spearman 和加权 Kendall τ（`kendall_weighted`）。不满足计算条件的指标写为空值，例如单样本组的相关系数。`target` 的单位及正负号应与训练数据的约定一致；程序不会自动转换。

目录中已有的 `outputs/` 是历史运行产物。复现或报告结果时，应结合对应的 `run_info.json` 核对权重哈希、配置和环境，并以本次运行结果为准。程序会覆盖本次生成的同名文件；切换输入数据时建议使用新的输出目录，避免混入旧结果。

## 使用自己的数据或权重

- 当前入口固定读取 `data/S669.csv` 和上述三路特征文件名。使用其他数据时，需要按此命名组织输入，或自行修改 `predict.py` 的路径映射。
- CSV 至少需要非空的 `ID` 列。`target` 可省略，此时仅输出预测和运行信息；若保留 `target`，该列必须全部为有限数值。
- `.pt` / `.pkl` 特征采用 `ID → 特征向量` 的字典格式。特征加载函数也支持含 `ID` 列的 CSV，但使用 CSV 文件时需同步调整入口中的文件路径。
- 特征必须遵循训练时的提取、预处理和列定义；提供原始序列不会触发自动 embedding 提取。
- 检查点必须包含 `state_dict`、`params`、`input_dims`、`feature_columns`、`scalers` 和 `routing_mode`，不能用只保存参数的 `state_dict` 替代完整检查点。
- 新权重应与训练时 scaler、模型配置和清单配套导出。不要通过删除清单来绕过不明原因的校验失败。

## 常见问题

| 问题 | 处理方式 |
| --- | --- |
| 找不到原来的 `/home/.../S669_inputs.zip` | 修改 notebook 的 `zip_path`，或按上文启用 Colab 上传模式 |
| `FileNotFoundError`，找不到权重或数据 | 检查是否已解压，以及 `--data-root` 是否指向 `weights/`、`data/` 的上一级目录 |
| 文件完整性校验失败 | 核对配套输入包；当前随附版本的已知权重清单问题见“核对当前输入包的权重清单” |
| 提示需要完整检查点 | 重新导出包含配置和 scaler 的检查点，单独的 `best_model.pth` 不满足 S669 入口要求 |
| 特征 ID、列或维度不匹配 | 检查三个分支的样本覆盖、特征列及其与训练流程的一致性 |
| 当前环境未提供 CUDA | 使用 `--device cpu` 或 `--device auto` |

## S571 迁移学习文件

`S571_inputs/` 包含 S571 的 ΔTm 数据、三路特征、`best_model.pth` 和 `DTm-transfer.py`。该脚本会执行训练、验证和重复实验，**不属于上述 S669 纯推理流程**。

该脚本需要额外提供训练集元数据和三路训练特征，不能仅凭本目录的 S571 测试文件直接完成迁移学习。可先查看参数：

```bash
python S571_inputs/DTm-transfer.py --help
```

主要参数为 `--train_x1/2/3`、`--train_meta`、`--test_x1/2/3`、`--test_meta`、`--pretrained`、`--nums` 和 `--output`。其中 `--nums` 默认 200，脚本会执行重复训练；运行前应根据实验需求设置。S571 的 `best_model.pth` 不应直接替换 S669 的完整检查点。
