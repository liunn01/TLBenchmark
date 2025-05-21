# --- START OF FILE vLLM-Performance-v1.py (Modified: Discard server logs, Unified filenames, CSV row order) ---

import os
import sys
import time
import signal
import subprocess
import requests
import threading
import argparse
from typing import List, Optional, Dict, Any
from datetime import datetime
import psutil
import random
import csv

# 强制使用非图形后端以避免在无头服务器上出错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 全局变量，用于在不同函数间共享命令行参数
args: Optional[argparse.Namespace] = None
# 全局 benchmark_summary_log_file 变量，以便 run_benchmark 可以访问 (用于记录客户端错误)
benchmark_summary_log_file: Optional[str] = None


# 函数：解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='vLLM GPU 性能基准测试工具',
        add_help=False
    )
    class CustomHelpAction(argparse.Action):
        def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
            super(CustomHelpAction, self).__init__(option_strings=option_strings, dest=dest, default=default, nargs=0, help=help)
        def __call__(self, parser, namespace, values, option_string=None):
            parser.print_help()
            print("\n--- 要查看 vLLM 服务器选项, 请运行: vllm serve --help ---")
            print("--- 要查看 vLLM 基准测试客户端选项, 请运行: vllm bench serve --help ---")
            parser.exit()
    parser.add_argument('-h', '--help', action=CustomHelpAction, help='显示此帮助信息并退出, 同时提示 vLLM 相关命令的帮助信息获取方式.')
    server_group = parser.add_argument_group('服务器配置 (传递给 "vllm serve")')
    server_group.add_argument('--model', type=str, required=True, help='模型的路径或 HuggingFace ID。此为必需参数。')
    server_group.add_argument('--host', type=str, default="0.0.0.0", help='vLLM 服务器运行的主机 (默认: %(default)s)')
    server_group.add_argument('--port', type=int, default=8335, help='vLLM 服务器运行的端口 (默认: %(default)s)')
    server_group.add_argument('--gpu', type=str, default="0", help='要使用的 GPU 设备 ID (逗号分隔) (默认: %(default)s)')
    server_group.add_argument('--tensor-parallel-size', '-tp', type=int, default=1, help='张量并行大小 (默认: %(default)s)')
    server_group.add_argument('--data-parallel-size', '-dp', type=int, default=None, help='数据并行大小。(默认: vLLM 默认)')
    server_group.add_argument('--max-num-batched-tokens', type=int, help='服务器最大批处理 token 数 (默认: vLLM 模型配置)', default=argparse.SUPPRESS)
    server_group.add_argument('--max-num-seqs', type=int, default=None, help='服务器批处理中最大序列数 (如果未指定，则使用 vLLM 服务器的默认值)')
    server_group.add_argument('--trust-remote-code', action='store_true', help='加载模型时信任远程代码 (默认: %(default)s)')
    server_group.add_argument('--enable-expert-parallel', action='store_true', help='为 MoE 模型启用专家并行 (默认: %(default)s)')
    server_group.add_argument('--no-enable-chunked-prefill', action='store_true', help='如果指定，则禁用 chunked prefill 功能。')
    server_group.add_argument('--no-enable-prefix-caching', action='store_true', help='如果指定，则禁用 prefix caching 功能。')

    client_group = parser.add_argument_group('基准测试客户端配置 (传递给 "vllm bench serve")')
    client_group.add_argument('--random-input-len', type=int, default=200, help='随机输入 token 的长度 (默认: %(default)s)')
    client_group.add_argument('--random-output-len', type=int, default=2000, help='请求的随机输出 token 长度 (默认: %(default)s)')
    client_group.add_argument(
        '--random-range-ratio',
        type=float,
        default=None,
        help='输入/输出长度的采样比例范围 (传递给 vllm bench serve)。如果未通过此脚本参数指定，则 vllm bench serve 将使用其默认值 (通常为 1.0)。'
    )
    client_group.add_argument('--concurrency-levels', type=str, default="4,8,16,32,64,128,256", help='要测试的并发级别 (逗号分隔) (默认: %(default)s)')
    client_group.add_argument('--prompts-multiplier', type=int, default=5, help='提示数量乘数 (默认: %(default)s)')

    general_group = parser.add_argument_group('通用脚本配置')
    general_group.add_argument('--log-dir', type=str, default="./benchmark_logs", help='所有日志和结果的存放目录 (默认: %(default)s)')

    parsed_args = parser.parse_args()
    if not parsed_args.model: parser.error("--model 参数是必需的。")
    parsed_args.inferred_log_model_name = os.path.basename(parsed_args.model).replace("/", "_").replace("\\", "_")
    print(f"提示: API服务名将由 'vllm serve' 从模型路径 '{parsed_args.model}' 推断。")
    print(f"提示: 'vllm bench serve' 将使用 '{parsed_args.model}' 作为其 --model 参数。")
    print(f"提示: 用于日志文件名的模型部分将是 '{parsed_args.inferred_log_model_name}'。")
    return parsed_args

# 类：进程管理器
class ProcessManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self._register_signals()

    def _register_signals(self):
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

    def start_process(self, cmd_list: List[str], log_file: Optional[str] = None, prefix: str = "", discard_output: bool = False) -> subprocess.Popen:
        if log_file and not discard_output:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        command_string = " ".join(cmd_list)
        stdout_dest = None
        stderr_dest = None

        if discard_output:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL
        elif log_file: 
            stdout_dest = subprocess.PIPE
            stderr_dest = subprocess.STDOUT
        
        proc = subprocess.Popen(
            command_string, shell=True, 
            stdout=stdout_dest, 
            stderr=stderr_dest,
            text=True, bufsize=1, encoding='utf-8', errors='replace'
        )
        self.processes.append(proc)
        log_message_suffix = ""
        if discard_output:
            log_message_suffix = "(输出已丢弃)"
        elif log_file:
            log_message_suffix = f"(日志记录到: {log_file})"
        print(f"已启动进程: PID={proc.pid}, 命令={command_string} {log_message_suffix}")

        if not discard_output and log_file and proc.stdout:
            log_thread = None
            def log_worker():
                try:
                    with open(log_file, "a", encoding='utf-8') as f:
                        if proc.stdout is None:
                             print(f"日志工作线程 '{prefix}' 警告: proc.stdout 为 None，无法读取日志。")
                             return
                        while True:
                            line = proc.stdout.readline()
                            if not line and proc.poll() is not None: break
                            if line:
                                output = f"[{prefix}] {line.strip()}"
                                print(output)
                                f.write(f"{datetime.now().isoformat()} {output}\n")
                            elif proc.poll() is not None: time.sleep(0.01)
                except Exception as e: print(f"日志工作线程 '{prefix}' 出错: {e}")
            
            log_thread = threading.Thread(target=log_worker, daemon=True)
            log_thread.start()
        
        return proc

    def cleanup(self, signum=None, frame=None):
        global args
        print("\n[清理] 正在终止进程...")
        cleanup_success = True
        for proc_obj in self.processes:
            try:
                if not proc_obj or proc_obj.poll() is not None: continue
                print(f"正在终止进程: PID={proc_obj.pid}")
                if proc_obj.stdout and not proc_obj.stdout.closed :
                    try: proc_obj.stdout.close()
                    except: pass
                proc_obj.terminate()
                try: proc_obj.wait(timeout=5); print(f"进程 {proc_obj.pid} 已成功终止。")
                except subprocess.TimeoutExpired:
                    print(f"进程 {proc_obj.pid} 未在5秒内终止，发送 SIGKILL...")
                    proc_obj.kill()
                    try: proc_obj.wait(timeout=2); print(f"进程 {proc_obj.pid} 已成功杀死。")
                    except subprocess.TimeoutExpired: print(f"错误: 未能杀死进程 {proc_obj.pid}。"); cleanup_success = False
            except (ProcessLookupError, AttributeError): pass
            except Exception as e: print(f"终止进程 {proc_obj.pid if proc_obj else 'N/A'} 时出错: {e}"); cleanup_success = False

        if args:
            try: self._kill_zombie_processes()
            except Exception as e: print(f"清理僵尸进程时出错: {e}"); cleanup_success = False
            try: self._kill_process_by_port(args.port)
            except Exception as e: print(f"通过端口清理进程时出错: {e}"); cleanup_success = False
            try: self._kill_gpu_processes()
            except Exception as e: print(f"清理 GPU 进程时出错: {e}"); cleanup_success = False
        else: print("[清理] 警告: 全局 'args' 未初始化，跳过部分 psutil 清理步骤。")
        status_message = "成功完成" if cleanup_success else "完成但有错误"
        print(f"[清理] {status_message}。")
        sys.exit(0 if cleanup_success else 1)

    def _kill_zombie_processes(self):
        zombie_count = 0; error_count = 0; current_pid = os.getpid(); script_name = os.path.basename(__file__)
        for proc_info_obj in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc_info_obj.info['pid'] == current_pid: continue
                if 'python' in proc_info_obj.info['name'].lower():
                    cmdline = proc_info_obj.info.get('cmdline')
                    if cmdline and ('vllm' in ' '.join(cmdline) or script_name in ' '.join(cmdline)):
                        print(f"强制杀死僵尸/相关 Python 进程: PID={proc_info_obj.info['pid']}, 命令={' '.join(cmdline)}")
                        psutil.Process(proc_info_obj.info['pid']).kill(); zombie_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): continue
            except Exception as e: print(f"杀死僵尸进程 PID {proc_info_obj.info.get('pid', 'N/A')} 时出错: {e}"); error_count += 1
        print(f"僵尸/相关进程清理: {zombie_count} 个进程被杀死, {error_count} 个错误。")

    def _kill_process_by_port(self, port: int):
        found_killed = 0; current_pid = os.getpid()
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    if conn.pid and conn.pid != current_pid:
                        try:
                            proc_to_kill = psutil.Process(conn.pid)
                            print(f"正在通过端口 {port} 杀死进程: PID={conn.pid}, 名称={proc_to_kill.name()}")
                            proc_to_kill.kill(); found_killed +=1; print(f"已成功杀死 PID {conn.pid} (端口 {port})。")
                        except psutil.NoSuchProcess: print(f"端口 {port} 上的进程 PID {conn.pid} 已不存在。")
                        except Exception as e: print(f"通过端口 {port} 杀死进程 PID {conn.pid} 时出错: {e}")
            if found_killed == 0: print(f"未发现在端口 {port} 上监听的活动进程可供杀死 (已排除当前脚本自身)。")
        except Exception as e: print(f"搜索/杀死端口 {port} 上的进程时失败: {e}")

    def _kill_vllm_main_process(self, port: int): pass

    def _kill_gpu_processes(self):
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            pids_str_list = output.decode().strip().split('\n')
            if not pids_str_list or not pids_str_list[0]: print("nvidia-smi 未报告活动的 GPU 进程。"); return
            killed_count = 0; error_count = 0; current_pid = os.getpid()
            for pid_str in pids_str_list:
                pid_str = pid_str.strip()
                if pid_str:
                    try:
                        pid_val = int(pid_str)
                        if pid_val == current_pid: continue
                        proc_gpu = psutil.Process(pid_val)
                        print(f"强制杀死 GPU 进程: PID={pid_val}, 名称={proc_gpu.name()}")
                        proc_gpu.kill(); killed_count += 1
                    except ValueError: print(f"来自 nvidia-smi 的 PID 格式无效: '{pid_str}'"); error_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied): error_count += 1
                    except Exception as e: print(f"杀死 GPU 进程 PID {pid_val} 时出错: {e}"); error_count += 1
            print(f"GPU 进程清理: {killed_count} 个进程被杀死, {error_count} 个错误。")
        except subprocess.CalledProcessError: print(f"nvidia-smi 命令失败。是否已安装并在 PATH 中?")
        except FileNotFoundError: print("未找到 nvidia-smi 命令。NVIDIA 驱动/CUDA 工具包可能未正确安装。")
        except Exception as e: print(f"杀死 GPU 进程时发生一般错误: {e}")

# 函数：等待服务器启动
def wait_for_server(host: str, port: int, timeout: int = 600) -> bool:
    print(f"正在等待服务器 {host}:{port} 启动...")
    start_time = time.time(); last_print_time = start_time
    health_url = f"http://{host}:{port}/health"; completions_url = f"http://{host}:{port}/v1/completions"
    while time.time() - start_time < timeout:
        health_ok, completions_active = False, False
        health_code, completions_code = "N/A", "N/A"
        try:
            hr = requests.get(health_url, timeout=2); health_code = hr.status_code
            if health_code == 200:
                health_ok = True
                try:
                    cr = requests.get(completions_url, timeout=2); completions_code = cr.status_code
                    if completions_code == 405: completions_active = True
                except: pass
            if health_ok and completions_active:
                print(f"\n服务器已就绪 (/health: {health_code}, /v1/completions: {completions_code})!"); return True
        except: pass
        if time.time() - last_print_time > 30: print(". ", end="", flush=True); last_print_time = time.time()
        time.sleep(2)
    print(f"\n服务器在 {host}:{port} 未在 {timeout} 秒内完全就绪 (health: {health_code}, completions: {completions_code})。"); return False

# 函数：执行基准测试 (客户端)
def run_benchmark(pm_unused: ProcessManager, config: dict, summary_log_for_errors: str):
    global args
    if not args or not args.model:
        print("错误 (run_benchmark): 全局 'args.model' (模型路径/ID) 未设置。")
        return {"MaxConcurrency": str(config['concurrency']), "Error": "args.model 未定义"}

    model_identifier_for_client = args.model

    bench_client_cmd_list = [
        "vllm", "bench", "serve",
        "--model", model_identifier_for_client,
        "--endpoint", "/v1/completions",
        "--host", config['host'],
        "--port", str(config['port']),
        "--dataset-name", "random",
        "--random-input-len", str(config['input_len']),
        "--random-output-len", str(config['output_len']),
        "--num-prompts", str(config['num_prompts']),
        "--seed", str(config['seed']),
        "--max-concurrency", str(config['concurrency'])
    ]
    if args.trust_remote_code:
        bench_client_cmd_list.append("--trust-remote-code")

    if args.random_range_ratio is not None:
        bench_client_cmd_list.extend(["--random-range-ratio", str(args.random_range_ratio)])

    grep_patterns = [
        "Maximum request concurrency:", r"Successful requests:\s+[0-9]+",
        r"Benchmark duration \(s\):\s+[0-9.]+", r"Request throughput \(req/s\):\s+[0-9.]+",
        r"Output token throughput \(tok/s\):\s+[0-9.]+",
        r"Total Token throughput \(tok/s\):\s+[0-9.]+",
        r"Median TTFT \(ms\):\s+[0-9.]+", r"Median TPOT \(ms\):\s+[0-9.]+"
    ]
    grep_cmd_str = "grep -E --color=never '" + "|".join(grep_patterns) + "'"
    env_vars_prefix = ""
    full_bench_client_command_str = env_vars_prefix + " ".join(bench_client_cmd_list) + " | " + grep_cmd_str

    print(f"正在为并发级别 {config['concurrency']} 运行基准测试客户端")
    print(f"客户端命令 (内部参数): {' '.join(bench_client_cmd_list)}")
    print(f"完整执行命令: {full_bench_client_command_str}")

    try:
        proc_bench = subprocess.Popen(
            full_bench_client_command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace'
        )
        stdout, stderr = proc_bench.communicate()

        if proc_bench.returncode != 0:
            error_msg = f"基准测试客户端失败 (退出码 {proc_bench.returncode})"
            detailed_error = stderr.strip() if stderr else "无详细错误信息"
            if detailed_error: error_msg += f": {detailed_error}"
            print(error_msg)
            if summary_log_for_errors:
                with open(summary_log_for_errors, "a", encoding='utf-8') as f_err_log:
                     f_err_log.write(f"\n--- Client Error (Concurrency: {config['concurrency']}) ---\n{detailed_error}\n------------------------------------\n")
            return {"MaxConcurrency": str(config['concurrency']), "Error": f"基准测试客户端失败: {detailed_error}"}

        if not stdout.strip():
            print("警告: 基准测试客户端在 grep 后返回空结果。")
            return {"MaxConcurrency": str(config['concurrency']), "Error": "基准测试客户端返回空结果"}

        results = parse_benchmark_output(stdout)
        if "MaxConcurrency" not in results:
            results["MaxConcurrency"] = str(config['concurrency'])
        return results
    except subprocess.SubprocessError as e:
        print(f"执行基准测试客户端时发生子进程错误: {e}"); return {"MaxConcurrency": str(config['concurrency']), "Error": str(e)}
    except Exception as e:
        print(f"执行基准测试客户端时出错 (并发级别={config['concurrency']}): {e}"); return {"MaxConcurrency": str(config['concurrency']), "Error": str(e)}

# 函数：解析基准测试输出 (使用英文键名)
def parse_benchmark_output(output: str) -> dict:
    results = {}
    patterns = {
        "MaxConcurrency": "Maximum request concurrency",
        "SuccessRequests": "Successful requests",
        "RequestThroughput": "Request throughput (req/s)",
        "OutputTokenThroughput": "Output token throughput (tok/s)",
        "TotalTokenThroughput": "Total Token throughput (tok/s)",
        "TTFT": "Median TTFT (ms)",
        "TPOT": "Median TPOT (ms)"
    }
    for line in output.splitlines():
        line = line.strip()
        for key, desc_part in patterns.items():
            if desc_part in line:
                try:
                    results[key] = line.split(":", 1)[1].strip()
                    break
                except IndexError:
                    print(f"警告: 解析行 '{line}' 失败，无法按 ':' 分割。")
    return results

# 函数：获取GPU信息
def get_gpu_config_string(gpu_arg: str) -> str:
    num_gpus = 0
    if gpu_arg:
        valid_gpu_ids = [gid for gid in gpu_arg.split(',') if gid.strip().isdigit()]
        num_gpus = len(valid_gpu_ids)

    if num_gpus == 0:
        print("警告: 未提供有效的GPU ID或GPU参数为空。GPU配置将设为 UnknownGPU*0。")
        return "UnknownGPU*0"

    gpu_name_short = "UnknownGPU"
    try:
        first_gpu_id_str = valid_gpu_ids[0]
        command = ["nvidia-smi", f"--query-gpu=name", f"--id={first_gpu_id_str}", "--format=csv,noheader,nounits"]
        result = subprocess.run(
            command,
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        gpu_name_full = result.stdout.strip()

        if not gpu_name_full:
            print("警告: nvidia-smi 返回了空的 GPU 名称。")
            gpu_name_short = "EmptyName"
        else:
            name_to_parse = gpu_name_full
            prefixes_to_remove = ["NVIDIA Corporation", "NVIDIA", "AMD", "Advanced Micro Devices, Inc."]
            for prefix in prefixes_to_remove:
                if name_to_parse.upper().startswith(prefix.upper()):
                    name_to_parse = name_to_parse[len(prefix):].strip()
            parts = name_to_parse.split()
            if parts:
                model_keywords = ["L4", "L20", "L40", "A30", "A40", "A100", "H100", "V100",
                                  "RTX", "TITAN", "QUADRO",
                                  "MI50", "MI100", "MI210", "MI250", "MI300"]
                found_specific_model = False
                for i, part in enumerate(parts):
                    current_part_cleaned = part
                    is_model_like = any(char.isdigit() for char in current_part_cleaned) or \
                                    any(current_part_cleaned.upper().startswith(kw) for kw in model_keywords)
                    if is_model_like:
                        gpu_name_short = part
                        if (part.upper() == "RTX" or part.upper().startswith("L")) and i + 1 < len(parts):
                            next_part = parts[i+1]
                            if next_part[0].isdigit() or (part.upper().startswith("L") and next_part.upper() == "S"):
                                gpu_name_short += "" + next_part
                        found_specific_model = True
                        break
                if not found_specific_model:
                    non_brand_parts = [p for p in parts if p.upper() not in [b.upper() for b in prefixes_to_remove]]
                    if non_brand_parts: gpu_name_short = non_brand_parts[0]
                    elif parts: gpu_name_short = parts[0]
                    else: gpu_name_short = "Unnamed"
            else:
                 gpu_name_short = gpu_name_full.split()[-1] if gpu_name_full.split() else "UnknownSplit"
    except FileNotFoundError: gpu_name_short = "NvidiaSMINotFound"
    except ValueError as ve: gpu_name_short = "InvalidGPU_ID"
    except subprocess.CalledProcessError as e:
        print(f"警告: 执行 nvidia-smi 失败 (返回码 {e.returncode})。GPU 型号将为未知。")
        if e.stdout: print(f"nvidia-smi stdout: {e.stdout.strip()}")
        if e.stderr: print(f"nvidia-smi stderr: {e.stderr.strip()}")
        gpu_name_short = "NvidiaSMIError"
    except Exception as e:
        print(f"警告: 获取 GPU 型号时发生未知错误: {e}。GPU 型号将为未知。"); import traceback; traceback.print_exc(); gpu_name_short = "GPUFetchError"
    return f"{gpu_name_short}*{num_gpus}"

# 函数：绘制基准测试结果图表
def plot_results(results_list, output_file="benchmark_results.png"):
    valid_results = [r for r in results_list if r and "Error" not in r]
    if not valid_results: print("没有有效的基准测试结果可供绘制。"); return
    try:
        results_by_concurrency = {}
        for r_item in valid_results:
            if "MaxConcurrency" in r_item and isinstance(r_item["MaxConcurrency"], str) and r_item["MaxConcurrency"].strip().isdigit():
                mc_val = int(r_item["MaxConcurrency"])
                if mc_val not in results_by_concurrency: results_by_concurrency[mc_val] = r_item
        sorted_concurrencies = sorted(results_by_concurrency.keys())
        if not sorted_concurrencies: print("未找到有效的整数并发级别用于绘图。"); return
        x_axis_for_plot = sorted_concurrencies
        def get_metric_for_plot(key):
            metric_values = []
            for mc_int in sorted_concurrencies:
                res_dict = results_by_concurrency.get(mc_int, {}); value_str = res_dict.get(key)
                is_numeric = False
                if isinstance(value_str, str):
                    temp_val = value_str.strip()
                    if temp_val.startswith('-'): temp_val = temp_val[1:]
                    if temp_val.replace('.', '', 1).isdigit(): is_numeric = True
                if is_numeric: metric_values.append(float(value_str))
                else: metric_values.append(None)
            return metric_values

        metrics_to_plot = {
            "RequestThroughput": ("ReqThr (req/s, Higher is Better)", "o", 0),
            "OutputTokenThroughput": ("output throughput (tok/s, Higher is Better)", "s", 1),
            "TotalTokenThroughput": ("TotalTokThr (tok/s, Higher is Better)", "^", 2),
            "TTFT": ("Median TTFT (ms, Lower is Better)", "x", 3),
            "TPOT": ("Median TPOT (ms, Lower is Better)", "d", 4)
        }
        metric_data_all = {key: get_metric_for_plot(key) for key in metrics_to_plot}
    except Exception as e: print(f"处理绘图数据时出错: {e}。跳过绘图。"); import traceback; traceback.print_exc(); return

    plt.figure(figsize=(14, 8)); plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.get_cmap('viridis', len(metrics_to_plot))
    all_y_values_for_ylim = []
    for key, (label, marker, color_idx) in metrics_to_plot.items():
        y_data = metric_data_all.get(key)
        if y_data is None: continue
        valid_x = [x for x, y_val in zip(x_axis_for_plot, y_data) if y_val is not None]
        valid_y = [y_val for y_val in y_data if y_val is not None]
        if valid_x and valid_y:
            plt.plot(valid_x, valid_y, label=label, marker=marker, linestyle='-', linewidth=2, color=colors(color_idx))
            all_y_values_for_ylim.extend(valid_y)
    plt.title("vLLM Benchmark Results vs Max Concurrency", fontsize=18, fontweight='bold')
    plt.xlabel("Max Concurrency (Parallel Requests)", fontsize=14); plt.ylabel("Metric Value", fontsize=14)
    if x_axis_for_plot: plt.xticks(x_axis_for_plot, fontsize=12)
    plt.yticks(fontsize=12)
    if all_y_values_for_ylim:
        min_y_val = min(all_y_values_for_ylim); max_y_val = max(all_y_values_for_ylim)
        y_padding = 0.1 * abs(max_y_val - min_y_val) if max_y_val != min_y_val else 0.1 * abs(max_y_val if max_y_val !=0 else 1.0)
        if y_padding == 0 : y_padding = max(1.0, abs(max_y_val * 0.1) if max_y_val != 0 else 1.0)
        bottom_y = min_y_val - y_padding
        top_y = max_y_val + y_padding
        if min_y_val >= 0 : bottom_y = max(0, bottom_y)
        if bottom_y >= top_y : top_y = bottom_y + y_padding
        plt.ylim(bottom=bottom_y, top=top_y)
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True); plt.grid(True, linestyle=":", alpha=0.7)
    try: plt.tight_layout(); plt.savefig(output_file, dpi=150); print(f"已将基准测试结果图表保存到 {output_file}")
    except Exception as e_plot: print(f"保存图表时出错: {e_plot}")
    plt.close()

# 主函数
def main():
    global args, benchmark_summary_log_file
    pm = None
    try:
        args = parse_arguments()
        log_model_name_part = args.inferred_log_model_name 

        benchmark_output_log_dir = os.path.join(args.log_dir, "benchmark_output_log")
        os.makedirs(benchmark_output_log_dir, exist_ok=True)
        pm = ProcessManager()

        current_time_date_part = datetime.now().strftime("%Y%m%d")
        base_filename_parts = [
            log_model_name_part,
            f"Input{args.random_input_len}",
            f"Output{args.random_output_len}",
            current_time_date_part
        ]
        base_filename = "-".join(base_filename_parts)

        csv_file_path = os.path.join(benchmark_output_log_dir, f"{base_filename}.csv")
        plot_file = os.path.join(benchmark_output_log_dir, f"{base_filename}.png")
        benchmark_summary_log_file = os.path.join(benchmark_output_log_dir, f"{base_filename}.log")
        
        print(f"统一的输出文件名前缀 (不含扩展名): {base_filename}")
        print(f"CSV 文件将保存到: {csv_file_path}")
        print(f"PNG 图表将保存到: {plot_file}")
        print(f"基准测试摘要日志将保存到: {benchmark_summary_log_file}")

        vllm_server_cmd_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu}"
        vllm_server_base_cmd = [
            "vllm", "serve", args.model, "--host", args.host, "--port", str(args.port),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
        ]
        if args.max_num_seqs is not None:
            vllm_server_base_cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
        if args.data_parallel_size is not None and args.data_parallel_size > 0:
            vllm_server_base_cmd.extend(["--data-parallel-size", str(args.data_parallel_size)])
        if hasattr(args, "max_num_batched_tokens"):
            vllm_server_base_cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
        if args.no_enable_chunked_prefill: vllm_server_base_cmd.append("--no-enable-chunked-prefill")
        if args.no_enable_prefix_caching: vllm_server_base_cmd.append("--no-enable-prefix-caching")
        if args.trust_remote_code: vllm_server_base_cmd.append("--trust-remote-code")
        if args.enable_expert_parallel: vllm_server_base_cmd.append("--enable-expert-parallel")

        full_vllm_server_cmd_list = [vllm_server_cmd_prefix] + vllm_server_base_cmd
        pm.start_process(full_vllm_server_cmd_list, log_file=None, prefix="vLLM_Server_Discarded", discard_output=True)

        if not wait_for_server(args.host, args.port, timeout=1200): print("错误: 服务器未能启动。正在退出。"); sys.exit(1)

        parsed_concurrency_levels = []
        try:
            parsed_concurrency_levels = [int(c.strip()) for c in args.concurrency_levels.split(',') if c.strip()]
            if not parsed_concurrency_levels: raise ValueError("并发级别列表为空。")
        except ValueError as e_conc: print(f"错误: 无效的并发级别 '{args.concurrency_levels}'。 {e_conc}"); sys.exit(1)

        csv_header_english = ["Model", "InputLen", "OutputLen", "GPUConfig", "MetricType"]
        for conc_level in parsed_concurrency_levels:
            csv_header_english.append(f"Conc{conc_level}")
        with open(csv_file_path, "w", newline="", encoding='utf-8') as f_csv:
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(csv_header_english)

        all_run_results_structured: Dict[str, Dict[int, Any]] = {
            "TPOT": {}, "TTFT": {}, "OutputTokenThroughput": {}
        }
        raw_results_for_plot = []
        benchmark_client_config_base = {"host": args.host, "port": args.port, "input_len": args.random_input_len, "output_len": args.random_output_len}

        for concurrency_level in parsed_concurrency_levels:
            print(f"\n{'='*40}\n正在为并发级别 {concurrency_level} 启动基准测试客户端\n{'='*40}")
            current_benchmark_config = benchmark_client_config_base.copy()
            current_benchmark_config['concurrency'] = concurrency_level
            current_benchmark_config['num_prompts'] = concurrency_level * args.prompts_multiplier
            current_benchmark_config['seed'] = random.randint(10000, 99999)
            print(f"客户端使用模型标识 (用于tokenizer和API请求): '{args.model}', 随机种子: {current_benchmark_config['seed']}")
            print(f"客户端提示数量: {current_benchmark_config['num_prompts']} ({args.prompts_multiplier}x 最大并发)")
            single_run_results = run_benchmark(pm, current_benchmark_config, benchmark_summary_log_file)
            raw_results_for_plot.append(single_run_results)
            if "Error" in single_run_results or not single_run_results:
                error_msg = single_run_results.get("Error", "客户端未返回数据")
                print(f"跳过记录并发级别 {concurrency_level} 的结果: {error_msg}")
                all_run_results_structured["TPOT"][concurrency_level] = "ERROR"
                all_run_results_structured["TTFT"][concurrency_level] = "ERROR"
                all_run_results_structured["OutputTokenThroughput"][concurrency_level] = "ERROR"
                continue
            all_run_results_structured["TPOT"][concurrency_level] = single_run_results.get("TPOT", "N/A")
            all_run_results_structured["TTFT"][concurrency_level] = single_run_results.get("TTFT", "N/A")
            all_run_results_structured["OutputTokenThroughput"][concurrency_level] = single_run_results.get("OutputTokenThroughput", "N/A")
            with open(benchmark_summary_log_file, "a", encoding='utf-8') as f_bench_log_data:
                 f_bench_log_data.write(f"--- Concurrency: {concurrency_level} ---\n")
                 for key, value in single_run_results.items():
                     f_bench_log_data.write(f"{key}: {value}\n")
                 f_bench_log_data.write("\n")

        gpu_config_str = get_gpu_config_string(args.gpu)
        model_name_for_csv = args.inferred_log_model_name
        with open(csv_file_path, "a", newline="", encoding='utf-8') as f_csv:
            csv_writer = csv.writer(f_csv)
            common_prefix = [ model_name_for_csv, args.random_input_len, args.random_output_len, gpu_config_str ]

            # --- 修改: 定义CSV行写入顺序 ---
            # 定义期望的指标顺序和它们在CSV中的标签
            ordered_metrics_for_csv = [
                ("OutputTokenThroughput", "output throughput"),
                ("TTFT", "Median TTFT"),
                ("TPOT", "Median TPOT")
            ]

            for internal_key, display_name in ordered_metrics_for_csv:
                if internal_key in all_run_results_structured:
                    concurrency_data = all_run_results_structured[internal_key]
                    row_to_write = common_prefix + [display_name]
                    for conc_level in parsed_concurrency_levels:
                        row_to_write.append(concurrency_data.get(conc_level, "N/A"))
                    csv_writer.writerow(row_to_write)
                else:
                    print(f"警告: 在 all_run_results_structured 中未找到内部键 '{internal_key}' 对应的数据。")
            # --- 结束修改 ---

        print(f"新的英文CSV格式结果已保存到: {csv_file_path}")
        if any("Error" not in r for r in raw_results_for_plot if r): plot_results(raw_results_for_plot, plot_file)
        else: print("没有成功的基准测试结果可供绘制。")
    except KeyboardInterrupt: print("\n基准测试被用户中断。")
    except SystemExit as e: print(f"脚本因 SystemExit 退出 (代码: {e.code})。");
    except Exception as e_main: print(f"\n[错误] main 函数中发生意外错误: {str(e_main)}"); import traceback; traceback.print_exc()
    finally:
        print("提示: main 函数的 finally 块: 如果 pm 已初始化，尝试清理...")
        if pm:
            try: pm.cleanup()
            except SystemExit as e_cleanup_exit:
                if e_cleanup_exit.code != 0: sys.exit(e_cleanup_exit.code)
            except Exception as e_final_cleanup: print(f"main 函数的 finally 块中最终清理时出错: {e_final_cleanup}"); sys.exit(1)
        else: print("提示: ProcessManager (pm) 未初始化，跳过 main 函数 finally 块中的清理调用。")

if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if e.code != 0: print(f"脚本最终以错误代码 {e.code} 退出。")
    except Exception as e_global:
        print(f"[严重错误] 全局范围发生未处理异常: {e_global}")
        import traceback; traceback.print_exc(); sys.exit(1)

# --- END OF FILE vLLM-Performance-v1.py (Modified: Discard server logs, Unified filenames, CSV row order) ---