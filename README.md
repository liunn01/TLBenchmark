# vLLM/NCCL/Baseinfo Benchmark Suite

**Version: 0.1.1**

## 简介

本项目用于自动化运行和汇总多种 GPU 相关的性能与环境信息测试，包括：
- **vLLM 推理性能测试（inference）**
- **NCCL 通信性能测试（nccl）**
- **主机与 GPU 环境信息采集（baseinfo）**

所有测试结果自动汇总为 Excel，日志和原始信息自动归档。

---

## 依赖环境

- Python 3.7+
- pandas
- openpyxl
- 相关测试依赖（如 nvidia-smi、nvcc、dmidecode、NCCL 测试工具等需在系统 PATH 下可用）

安装依赖：
```bash
pip install -r requirements-inference.txt
```

---

## 命令行参数说明

### 通用参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--log-dir` |      | 日志和结果输出目录 | `./benchmark_logs` |
| `-h, --help` |      | 显示帮助信息 |  |

---

### Inference Benchmark

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--inference` |      | 启用推理测试 |  |
| `--model` | `-m` | vLLM 模型标识（必填） |  |
| `--served-model-name` | `-n` | vLLM 服务模型名（必填） |  |
| `--host` | `-H` | vLLM 服务主机 | `localhost` |
| `--port` | `-P` | vLLM 服务端口 | `8000` |
| `--random-input-len` | `-i` | 随机输入 token 长度，支持逗号分隔多组 | `200` |
| `--random-output-len` | `-o` | 随机输出 token 长度，支持逗号分隔多组 | `2000` |
| `--concurrency-levels` | `-c` | 并发级别，支持逗号分隔或多组如`[2,4,8],[16,32]` | `4,8,16,32,64,128` |
| `--trust-remote-code` | `-t` | 信任远程代码 |  |
| `--vbs-prompts-multiplier` | `-M` | prompts 乘数 | `5` |
| `--vbs-random-range-ratio` | `-r` | 随机范围比 |  |
| `--skip-server-check` | `-s` | 跳过服务可达性检查 |  |
| `--server-gpu-info-override` | `-g` | 手动指定 GPU 型号 |  |

#### 多组参数 sweep 说明

- 支持如下格式批量测试多组参数（按顺序一一配对）：
  ```
  --random-input-len "100,200" --random-output-len "1000,2000" --concurrency-levels "[2,4,8],[16,32]"
  ```
  会顺序执行两组参数，结果自动合并汇总。

---

### NCCL Benchmark

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--nccl` |      | 启用 NCCL 测试 |  |
| `--BandWidth-Test` | `-b` | bandwidthTest 命令路径 | `./testtools/bandwidthTest` |
| `--p2p-test` | `-p` | p2pBandwidthLatencyTest 命令路径 | `./testtools/p2pBandwidthLatencyTest` |
| `--All_Reduce_perf` | `-R` | all_reduce_perf 命令路径 | `./testtools/all_reduce_perf` |
| `--All_Gather_perf` | `-G` | all_gather_perf 命令路径 | `./testtools/all_gather_perf` |

> **注意：** 这几个参数只指定可执行文件路径，实际执行时会自动拼接原有参数。

---

### Baseinfo

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--baseinfo` |      | 采集主机与 GPU 环境信息 |  |

---

## 用法与示例

### 1. 只采集主机与 GPU 环境信息

```bash
python trngt.py --baseinfo
```

### 2. 只跑推理

```bash
python trngt.py --inference -m /local/Qwen2-7B -n qwen7b -H 0.0.0.0 -P 8000 -i "20,200" -o "200,20" -c "[8,16,32,64,128],[2,4,6,8]" -t -M 2 -g "TL-100*1"
```

### 3. 只跑 NCCL（可选自定义命令路径）

```bash
python trngt.py --nccl
# 或自定义命令路径：
python trngt.py --nccl -b /my/path/bandwidthTest -B /my/path/p2pBandwidthLatencyTest -R /my/path/all_reduce_perf -G /my/path/all_gather_perf
```

### 4. 所有功能都测试（推荐示例）

```bash
python trngt.py --inference -m /local/Qwen2-7B -n qwen7b -H 0.0.0.0 -P 8000 -i "20,200" -o "200,20" -c "[8,16,32,64,128],[2,4,6,8]" -t -M 2 -g "TL-100*1" --nccl --baseinfo --log-dir=./test2/
```

### 5. 自定义日志目录

```bash
python trngt.py --inference --log-dir ./mylogs ...
```

---

## 结果输出说明

- **每个功能的结果**会分别输出到 `--log-dir` 指定的子目录下（默认为 `./benchmark_logs`）。
- **inference**：每组参数生成一个 csv，所有推理结果自动合并到 summary_时间戳.xlsx 的 inference sheet，并在终端只输出一次总表格。
- **nccl/baseinfo**：各自输出 csv/xlsx 到对应子目录，最终也合并到 summary_时间戳.xlsx 的对应 sheet。
- **summary_时间戳.xlsx**：本次所有新生成的结果自动合并，每个功能一个 sheet（inference、nccl、baseinfo）。

---

## 其它说明

- 日志和结果文件均带有时间戳，便于多次运行归档。
- 支持多组推理参数 sweep，自动顺序执行并合并结果。
- NCCL 测试依赖外部二进制，需提前编译好 CUDA Samples 和 NCCL Tests。
- 只汇总本次新生成的结果文件，不会混入历史文件。
- 终端只输出一次总的推理 summary 表格。

---

如有问题请联系维护者。 
