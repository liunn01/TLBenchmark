# --- START OF FILE vllmtest.py (CSV GPU column fix for N/A, Plot and CSV label changes, CSV Header concise) ---

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
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args_global: Optional[argparse.Namespace] = None
benchmark_summary_log_file_global: Optional[str] = None

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='vLLM GPU Performance Benchmark Client Tool (Connects to an existing server)',
        add_help=False
    )
    class CustomHelpAction(argparse.Action):
        def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
            super(CustomHelpAction, self).__init__(option_strings=option_strings, dest=dest, default=default, nargs=0, help=help)
        def __call__(self, parser, namespace, values, option_string=None):
            parser.print_help()
            client_help_command = "vllm bench serve --help"
            print(f"\n--- To see benchmark client options, run: {client_help_command} ---")
            print("--- Note: This script does NOT start the vLLM server. Ensure the server is already running. ---")
            parser.exit()

    parser.add_argument('-h', '--help', action=CustomHelpAction, help='Show this help message and exit.')

    server_conn_group = parser.add_argument_group('Target Server Configuration')
    server_conn_group.add_argument('--model', type=str, required=True, help='Model identifier for vLLM client commands. This is required.')
    server_conn_group.add_argument('--host', type=str, default="localhost", help='Host where the target vLLM server is running (default: %(default)s)')
    server_conn_group.add_argument('--port', type=int, default=8000, help='Port where the target vLLM server is running (default: %(default)s)')

    client_params_group = parser.add_argument_group('Common Client Parameters (passed to "vllm bench serve")')
    client_params_group.add_argument('--random-input-len', type=int, default=200, help='Length of random input tokens (default: %(default)s)')
    client_params_group.add_argument('--random-output-len', type=int, default=2000, help='Requested length of random output tokens (default: %(default)s)')
    client_params_group.add_argument('--concurrency-levels', type=str, default="4,8,16,32,64,128", help='Comma-separated concurrency levels to test (default: %(default)s)')
    client_params_group.add_argument('--trust-remote-code', action='store_true', help='Trust remote code when loading the model if required by the client command (default: %(default)s)')

    vbs_client_group = parser.add_argument_group('vllm bench serve Client-Specific Parameters')
    vbs_client_group.add_argument('--vbs-prompts-multiplier', type=int, default=5, help='(vllm bench serve) Number of prompts multiplier (default: %(default)s)')
    vbs_client_group.add_argument('--vbs-random-range-ratio', type=float, default=None, help='(vllm bench serve) --random-range-ratio (default: vLLM default)')

    general_group = parser.add_argument_group('General Script Configuration')
    general_group.add_argument('--log-dir', type=str, default="./benchmark_logs_client_only", help='Directory for all logs and results (default: %(default)s)')
    general_group.add_argument('--skip-server-check', action='store_true', help='Skip initial server reachability check.')
    general_group.add_argument('--server-gpu-info-override', type=str, default=None, help='Manually override server GPU model (e.g., "NVIDIA RTX 4090"). Skips auto-detection if provided.')

    parsed_args = parser.parse_args()
    if not parsed_args.model: parser.error("--model argument is required.")

    parsed_args.inferred_log_model_name = os.path.basename(parsed_args.model).replace("/", "_").replace("\\", "_")
    print(f"Client will connect to server at {parsed_args.host}:{parsed_args.port}.")
    print(f"Client will use model identifier: '{parsed_args.model}' (for vllm bench serve command).")
    return parsed_args

class ProcessManager:
    def __init__(self):
        self._register_signals()

    def _register_signals(self):
        signal.signal(signal.SIGINT, self._handle_exit_signal)
        signal.signal(signal.SIGTERM, self._handle_exit_signal)

    def _handle_exit_signal(self, signum, frame):
        print(f"\n[Cleanup] Received signal {signal.Signals(signum).name}, attempting cleanup and exit...")
        self._cleanup_script_launched_processes()
        print("[Cleanup] Client script exiting.")
        sys.exit(128 + signum)

    def _cleanup_script_launched_processes(self):
        print("[Cleanup] Checking for potential zombie processes launched by this script...")
        current_pid = os.getpid()
        killed_count = 0
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
                if proc.info['pid'] == current_pid: continue
                is_child_of_script = proc.info.get('ppid') == current_pid
                cmdline_list = proc.info.get('cmdline', [])
                if not cmdline_list: continue
                cmdline_str_lower = ' '.join(cmdline_list).lower()
                is_vllm_bench_serve_like = "vllm" in cmdline_str_lower and "bench" in cmdline_str_lower and "serve" in cmdline_str_lower

                if is_child_of_script and is_vllm_bench_serve_like:
                    try:
                        p_obj = psutil.Process(proc.info['pid'])
                        print(f"[Cleanup] Found zombie client process launched by this script: PID={p_obj.pid}, Command='{' '.join(cmdline_list)}'. Attempting to terminate...")
                        p_obj.terminate()
                        try: p_obj.wait(timeout=1.0)
                        except psutil.TimeoutExpired:
                            print(f"[Cleanup] Process PID={p_obj.pid} did not terminate in 1s, killing...")
                            p_obj.kill()
                        killed_count +=1
                        print(f"[Cleanup] Process PID={p_obj.pid} handled.")
                    except psutil.NoSuchProcess: pass
                    except Exception as e_kill: print(f"[Cleanup] Error terminating process PID={proc.info['pid']}: {e_kill}")
            if killed_count > 0: print(f"[Cleanup] {killed_count} potential zombie client processes were attempted to be terminated.")
            else: print("[Cleanup] No zombie client processes launched by this script found needing cleanup.")
        except Exception as e: print(f"[Cleanup] Error during cleanup of script-launched processes: {e}")

    def cleanup_at_exit(self):
        print("\n[Cleanup] Script execution finished, performing final cleanup...")
        self._cleanup_script_launched_processes()
        print("[Cleanup] Cleanup complete.")

def check_server_availability(host: str, port: int, timeout: int = 10) -> bool:
    print(f"Checking server reachability at {host}:{port}...")
    check_urls_responses = {
        f"http://{host}:{port}/health": [200],
        f"http://{host}:{port}/v1/models": [200],
        f"http://{host}:{port}/v1/completions": [405] # Method Not Allowed is fine for a GET
    }
    for url_to_check, expected_codes in check_urls_responses.items():
        try:
            response = requests.get(url_to_check, timeout=timeout)
            if response.status_code in expected_codes:
                print(f"Server responded normally at {url_to_check} (Status: {response.status_code}).")
                return True
            else:
                print(f"Server responded unexpectedly at {url_to_check} (Status: {response.status_code}, Expected: {expected_codes}).")
        except requests.exceptions.ConnectionError: print(f"Could not connect to server at {url_to_check}.")
        except requests.exceptions.Timeout: print(f"Connection to server at {url_to_check} timed out.")
        except requests.exceptions.RequestException as e: print(f"Error checking server at {url_to_check}: {e}")
    print(f"Failed to confirm server reachability at {host}:{port} via any known endpoint. Please ensure the server is running and listening.")
    return False

def get_vllm_server_gpu_info_local(server_host: str, server_port: int) -> str:
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        print("Warning: nvidia-smi command not found locally.")
        return "N/A_local_nvidia-smi_not_found"

    try:
        cmd = [nvidia_smi_path, "-q", "-i", "0"] # Query GPU 0
        print(f"Executing command to get GPU model: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        
        output_lines = result.stdout.strip().splitlines()
        for line in output_lines:
            if "Product Name" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    gpu_model = parts[1].strip()
                    if gpu_model:
                        print(f"Successfully extracted GPU model via nvidia-smi -q -i 0: {gpu_model}")
                        return gpu_model
        
        print(f"Warning: Could not parse 'Product Name' from 'nvidia-smi -q -i 0' output.")
        print(f"nvidia-smi output (first 15 lines):\n" + "\n".join(output_lines[:15]))
        return "N/A_local_smi_q_parse_error"

    except subprocess.CalledProcessError as e:
        print(f"Warning: 'nvidia-smi -q -i 0' command failed. Return code: {e.returncode}")
        if e.stdout: print(f"Stdout: {e.stdout.strip()}")
        if e.stderr: print(f"Stderr: {e.stderr.strip()}")
        return f"N/A_local_smi_q_cmd_error_{e.returncode}"
    except subprocess.TimeoutExpired:
        print("Warning: 'nvidia-smi -q -i 0' command timed out.")
        return "N/A_local_smi_q_cmd_timeout"
    except FileNotFoundError: 
        print("Warning: nvidia-smi command not found locally (FileNotFoundError).")
        return "N/A_local_nvidia-smi_not_found"
    except Exception as e:
        import traceback
        print(f"Unknown error getting local server GPU info (nvidia-smi -q -i 0): {e}\n{traceback.format_exc()}")
        return "N/A_local_smi_q_unknown_error"
    return "N/A_local_smi_q_fallback"


def run_benchmark(config: dict, summary_log_for_client: str):
    global args_global
    if not args_global or not args_global.model:
        print("Error (run_benchmark): Global 'args_global.model' not set.")
        return {"MaxConcurrency": str(config.get('current_concurrency_level', 'N/A_Conf')), "Error": "args_global.model not defined"}

    bench_client_cmd_list = [
        "vllm", "bench", "serve",
        "--model", args_global.model, "--endpoint", "/v1/completions",
        "--host", config['host'], "--port", str(config['port']),
        "--dataset-name", "random", "--random-input-len", str(config['input_len']),
        "--random-output-len", str(config['output_len']),
        "--num-prompts", str(config['num_prompts_for_vbs']),
        "--seed", str(config['seed']), "--max-concurrency", str(config['current_concurrency_level'])
    ]
    if args_global.trust_remote_code: bench_client_cmd_list.append("--trust-remote-code")
    if args_global.vbs_random_range_ratio is not None:
        bench_client_cmd_list.extend(["--random-range-ratio", str(args_global.vbs_random_range_ratio)])

    print(f"Running benchmark client (vllm bench serve) for concurrency level {config['current_concurrency_level']}")
    command_to_run_str = " ".join(bench_client_cmd_list)
    print(f"Client command: {command_to_run_str}")

    full_stdout, full_stderr = "", ""
    proc_return_code = -1
    results = {}

    try:
        proc_bench = subprocess.Popen(
            command_to_run_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace'
        )
        full_stdout, full_stderr = proc_bench.communicate()
        proc_return_code = proc_bench.returncode

        if summary_log_for_client:
            with open(summary_log_for_client, "a", encoding='utf-8') as f_log:
                f_log.write(f"\n--- Client Output for Concurrency: {config['current_concurrency_level']} ---\n")
                f_log.write(f"Command: {command_to_run_str}\n")
                f_log.write(f"Return Code: {proc_return_code}\n")
                f_log.write(full_stdout if full_stdout.strip() else "<No stdout or stdout is empty>\n")
                if full_stderr.strip():
                    f_log.write("--- STDERR ---\n")
                    f_log.write(full_stderr)
                f_log.write("--- End Client Output ---\n")

        if proc_return_code != 0:
            error_msg = f"vllm bench serve execution failed (exit code {proc_return_code})"
            if full_stderr.strip(): error_msg += f". Stderr: {full_stderr.strip()}"
            elif not full_stdout.strip(): error_msg += ". Stdout was also empty."
            print(error_msg)
            return {"MaxConcurrency": str(config['current_concurrency_level']), "Error": error_msg}

        if not full_stdout.strip():
            warning_msg = f"Warning: vllm bench serve returned empty output (Concurrency: {config['current_concurrency_level']})."
            print(warning_msg)
            return {"MaxConcurrency": str(config['current_concurrency_level']), "Error": "Client returned empty output"}

        results = parse_benchmark_output(full_stdout, config['current_concurrency_level'])
        return results

    except subprocess.SubprocessError as e: err_msg = f"Subprocess error during client execution: {e}"
    except Exception as e:
        import traceback
        err_msg = f"Unknown error during client execution: {e}\n{traceback.format_exc()}"

    print(err_msg)
    if summary_log_for_client:
        with open(summary_log_for_client, "a", encoding='utf-8') as f_log:
            f_log.write(f"\n--- Exception during client execution (Concurrency: {config['current_concurrency_level']}) ---\n{err_msg}\n")
    return {"MaxConcurrency": str(config['current_concurrency_level']), "Error": err_msg}


def parse_benchmark_output(output: str, concurrency_level_from_config: int) -> dict:
    results = {"MaxConcurrency": str(concurrency_level_from_config)}
    patterns = {
        "ParsedMaxConcurrency": r"Maximum request concurrency:\s*([0-9]+)",
        "SuccessRequests": r"Successful requests:\s*([0-9]+)",
        "RequestThroughput": r"Request throughput \(req/s\):\s*([0-9.]+)", 
        "OutputTokenThroughput": r"Output token throughput \(tok/s\):\s*([0-9.]+)",
        "TotalTokenThroughput": r"Total Token throughput \(tok/s\):\s*([0-9.]+)", 
        "TTFT": r"Median TTFT \(ms\):\s*([0-9.]+)",
        "TPOT": r"Median TPOT \(ms\):\s*([0-9.]+)"
    }
    expected_keys_for_structure = list(patterns.keys())

    for key, pattern_str in patterns.items():
        match = re.search(pattern_str, output, re.MULTILINE)
        if match:
            try: results[key] = match.group(1).strip()
            except IndexError: print(f"Warning: Failed to parse metric '{key}' using pattern '{pattern_str}'.")
    for key in expected_keys_for_structure:
        if key not in results:
            results[key] = "N/A"
    if "ParsedMaxConcurrency" in results and results["ParsedMaxConcurrency"] != "N/A":
        if results["ParsedMaxConcurrency"] != str(concurrency_level_from_config):
            print(f"Warning: Configured concurrency ({concurrency_level_from_config}) does not match parsed concurrency ({results['ParsedMaxConcurrency']}).")
    return results


def get_gpu_model_from_system(): 
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None: return "N/A_client_smi_not_found"
    try:
        result = subprocess.check_output([nvidia_smi_path, '--query-gpu=gpu_name', '--format=csv,noheader,nounits'], text=True, encoding='utf-8', stderr=subprocess.PIPE)
        gpu_models = result.strip().splitlines()
        if gpu_models: return gpu_models[0]
        return "N/A_client_no_gpu_info"
    except: return "N/A_client_smi_error"


def plot_results(results_list, output_file="benchmark_results.png"):
    global args_global
    valid_results = [r for r in results_list if r and "Error" not in r and "MaxConcurrency" in r]
    if not valid_results: print("No valid benchmark results to plot."); return

    client_mode_str = "vllm bench serve"
    try:
        results_by_concurrency = {}
        for r_item in valid_results:
            mc_str = r_item.get("MaxConcurrency")
            if isinstance(mc_str, str) and mc_str.strip().isdigit():
                mc_val = int(mc_str)
                results_by_concurrency[mc_val] = r_item
        if not results_by_concurrency: print("No valid results with numeric concurrency levels found for plotting."); return
        sorted_concurrencies = sorted(results_by_concurrency.keys())
        x_axis_for_plot = sorted_concurrencies

        def get_metric_for_plot(key_to_plot):
            metric_values = []
            for mc_int_val in sorted_concurrencies:
                res_dict_for_mc = results_by_concurrency.get(mc_int_val, {})
                value_str_or_num = res_dict_for_mc.get(key_to_plot)
                numeric_val = None
                if isinstance(value_str_or_num, (int, float)): numeric_val = float(value_str_or_num)
                elif isinstance(value_str_or_num, str):
                    temp_val_str = value_str_or_num.strip()
                    if temp_val_str and temp_val_str.lower() not in ["n/a", "error"]:
                        try: numeric_val = float(temp_val_str)
                        except ValueError: pass
                metric_values.append(numeric_val)
            return metric_values

        metrics_to_plot_config = {
            "OutputTokenThroughput": ("Output Token Throughput (tok/s, higher is better)", "s", 0),
            "TTFT": ("Median TTFT (ms, lower is better)", "x", 1),
            "TPOT": ("Median TPOT (ms, lower is better)", "d", 2)
        }
        metric_data_for_plotting = {key: get_metric_for_plot(key) for key in metrics_to_plot_config}
    except Exception as e: print(f"Error processing data for plotting: {e}. Skipping plot generation."); import traceback; traceback.print_exc(); return

    plt.figure(figsize=(16, 10)); plt.style.use('seaborn-v0_8-whitegrid')
    try: colors_palette = plt.cm.get_cmap('tab10', len(metrics_to_plot_config))
    except: colors_palette = plt.cm.get_cmap('viridis', len(metrics_to_plot_config))

    all_plotted_y_values = []
    for metric_key, (plot_label, marker_style, color_idx) in metrics_to_plot_config.items():
        y_values_list = metric_data_for_plotting.get(metric_key)
        if y_values_list is None: continue
        valid_x_points = [x for x, y_val in zip(x_axis_for_plot, y_values_list) if y_val is not None]
        valid_y_points = [y_val for y_val in y_values_list if y_val is not None]
        if valid_x_points and valid_y_points:
            plt.plot(valid_x_points, valid_y_points, label=plot_label, marker=marker_style,
                     linestyle='-', linewidth=2, markersize=8, color=colors_palette(color_idx % colors_palette.N))
            all_plotted_y_values.extend(valid_y_points)

    title_lines = [
        f"Benchmark Client: {client_mode_str}",
        f"Model (Client): {args_global.inferred_log_model_name}",
        f"Prompts: Input {args_global.random_input_len} / Output {args_global.random_output_len} tokens"
    ]
    server_gpu_display = args_global.server_gpu_info_override if args_global.server_gpu_info_override else getattr(args_global, '_detected_server_gpu_info', 'N/A_unknown_server_GPU')
    if server_gpu_display and server_gpu_display != "N/A" and not server_gpu_display.startswith("N/A_") :
        title_lines.insert(2, f"Server GPU (Detected/Provided): {server_gpu_display}")

    title_lines.append(f"Client Config: PromptsMultiplier={args_global.vbs_prompts_multiplier}" + \
                        (f", RandRatio={args_global.vbs_random_range_ratio}" if args_global.vbs_random_range_ratio is not None else ""))
    
    plt.title("\n".join(title_lines), fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Concurrency Level (Client --max-concurrency)", fontsize=14, labelpad=10)
    plt.ylabel("Metric Value", fontsize=14, labelpad=10)
    if x_axis_for_plot: plt.xticks(x_axis_for_plot, [str(x) for x in x_axis_for_plot], fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12)

    if all_plotted_y_values:
        numeric_y_values = [y for y in all_plotted_y_values if isinstance(y, (int, float))]
        if numeric_y_values:
            min_y_val = min(numeric_y_values)
            max_y_val = max(numeric_y_values)
            y_range = max_y_val - min_y_val
            y_padding = y_range * 0.1 if y_range > 0 else max(1.0, abs(max_y_val * 0.1))
            if y_padding == 0 and max_y_val == 0 : y_padding = 1.0
            y_axis_bottom = min_y_val - y_padding
            y_axis_top = max_y_val + y_padding
            if all(y >= 0 for y in numeric_y_values) and min_y_val < (y_axis_top * 0.1 if y_axis_top != 0 else 0.1):
                 y_axis_bottom = 0
            if y_axis_bottom >= y_axis_top : y_axis_top = y_axis_bottom + max(1.0, abs(y_axis_bottom * 0.1 if y_axis_bottom !=0 else 0.1))
            plt.ylim(bottom=y_axis_bottom, top=y_axis_top)

    plt.legend(fontsize=11, loc='best', frameon=True, shadow=True, title="Metrics Legend", title_fontsize='13')
    plt.grid(True, linestyle="--", alpha=0.6)
    try:
        plt.tight_layout(pad=1.5); plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Benchmark results chart saved to {output_file}")
    except Exception as e_plot: print(f"Error saving chart: {e_plot}")
    plt.close()

def main():
    global args_global, benchmark_summary_log_file_global
    pm = ProcessManager()
    exit_code = 0

    try:
        args_global = parse_arguments()
        log_model_name_part = args_global.inferred_log_model_name
        client_mode_str_for_log = "vbs"
        benchmark_output_log_dir = os.path.join(args_global.log_dir, f"bm_out_{client_mode_str_for_log}")
        os.makedirs(benchmark_output_log_dir, exist_ok=True)

        current_time_full_ts_for_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_filename_parts = [
            log_model_name_part,
            f"in{args_global.random_input_len}",
            f"out{args_global.random_output_len}",
            current_time_full_ts_for_filename
        ]
        base_filename = "-".join(base_filename_parts)

        csv_file_path = os.path.join(benchmark_output_log_dir, f"{base_filename}.csv")
        plot_file = os.path.join(benchmark_output_log_dir, f"{base_filename}.png")
        benchmark_summary_log_file_global = os.path.join(benchmark_output_log_dir, f"{base_filename}.log")

        if not args_global.skip_server_check:
            if not check_server_availability(args_global.host, args_global.port):
                print(f"Error: Target server at {args_global.host}:{args_global.port} is not reachable. Exiting.")
                sys.exit(1)
            else: print(f"Successfully connected to target server {args_global.host}:{args_global.port}.")
        else: print("Info: Server reachability check skipped.")

        parsed_concurrency_levels = []
        try:
            raw_levels = [c.strip() for c in args_global.concurrency_levels.split(',') if c.strip()]
            parsed_concurrency_levels = sorted(list(set([int(lvl) for lvl in raw_levels if lvl.isdigit()])))
            if not parsed_concurrency_levels: raise ValueError("Concurrency levels list is empty or invalid.")
        except ValueError as e_conc: print(f"Error: Invalid concurrency levels '{args_global.concurrency_levels}'. {e_conc}"); sys.exit(1)

        # --- CSV Header Change HERE ---
        csv_header_final = ["Model", "Input-len", "Output-len", "GPU", "Metric"] # Changed "ServerGPUModel" to "GPU"
        for conc_level in parsed_concurrency_levels: 
            csv_header_final.append(f"Conc{conc_level}") # Changed "Concurrency" to "Conc"
        # --- End CSV Header Change ---
        with open(csv_file_path, "w", newline="", encoding='utf-8') as f_csv:
            csv.writer(f_csv).writerow(csv_header_final)

        all_run_results_structured: Dict[str, Dict[int, Any]] = {
            k: {} for k in ["TPOT", "TTFT", "OutputTokenThroughput", "RequestThroughput", "TotalTokenThroughput", "ParsedMaxConcurrency", "SuccessRequests"]
        }
        raw_results_for_plot = []
        benchmark_client_config_base = { "host": args_global.host, "port": args_global.port,
            "input_len": args_global.random_input_len, "output_len": args_global.random_output_len }

        for concurrency_level_val in parsed_concurrency_levels:
            print(f"\n{'='*40}\nTesting Concurrency Level {concurrency_level_val}\n{'='*40}")
            current_benchmark_config = benchmark_client_config_base.copy()
            current_benchmark_config['current_concurrency_level'] = concurrency_level_val
            current_benchmark_config['seed'] = random.randint(10000, 99999)
            current_benchmark_config['num_prompts_for_vbs'] = concurrency_level_val * args_global.vbs_prompts_multiplier
            print(f"vllm bench serve: --max-concurrency={concurrency_level_val}, --num-prompts={current_benchmark_config['num_prompts_for_vbs']}" + \
                  (f", --random-range-ratio={args_global.vbs_random_range_ratio}" if args_global.vbs_random_range_ratio is not None else ""))

            single_run_results = run_benchmark(current_benchmark_config, benchmark_summary_log_file_global)
            raw_results_for_plot.append(single_run_results.copy()) 

            if "Error" in single_run_results or not single_run_results:
                error_msg = single_run_results.get("Error", "Client returned no data")
                print(f"Skipping result recording for concurrency {concurrency_level_val}: {error_msg}")
                for k_metric in all_run_results_structured: all_run_results_structured[k_metric][concurrency_level_val] = "ERROR"
                continue
            for k_metric_res in all_run_results_structured:
                 all_run_results_structured[k_metric_res][concurrency_level_val] = single_run_results.get(k_metric_res, "N/A")
            
        server_gpu_model_str = "N/A"
        if args_global.server_gpu_info_override:
            server_gpu_model_str = args_global.server_gpu_info_override
            print(f"Using user-provided server GPU model: {server_gpu_model_str}")
        elif args_global.host.lower() in ["localhost", "127.0.0.1", "0.0.0.0"] or not args_global.host:
            print(f"Attempting to auto-detect GPU model for local server (host: {args_global.host}, port: {args_global.port})...")
            detected_info = get_vllm_server_gpu_info_local(args_global.host, args_global.port)
            if detected_info and not detected_info.startswith("N/A_"):
                server_gpu_model_str = detected_info
            else:
                print(f"Warning: Auto-detection of server GPU model failed (Reason: {detected_info if detected_info else 'Unknown'}). GPU column will be set to N/A.")
                server_gpu_model_str = "N/A"
        else:
            print(f"Server host '{args_global.host}' is not local, cannot auto-detect GPU. Use --server-gpu-info-override. GPU column will be N/A.")
            server_gpu_model_str = "N/A"
        
        args_global._detected_server_gpu_info = server_gpu_model_str 
        print(f"Server GPU model for CSV (GPU column): {server_gpu_model_str}")

        model_name_for_csv = args_global.inferred_log_model_name
        base_data = [model_name_for_csv, str(args_global.random_input_len), str(args_global.random_output_len), server_gpu_model_str]

        target_metrics_csv = [("output throughput", "OutputTokenThroughput"),
                              ("Median TTFT", "TTFT"),
                              ("Median TPOT", "TPOT")]
                              
        with open(csv_file_path, "a", newline="", encoding='utf-8') as f_csv_append:
            writer = csv.writer(f_csv_append)
            for display_name, internal_key in target_metrics_csv:
                if internal_key not in all_run_results_structured: continue
                row_data_for_metric = all_run_results_structured[internal_key]
                row = base_data + [display_name] + [row_data_for_metric.get(c, "N/A") for c in parsed_concurrency_levels]
                writer.writerow(row)
        print(f"CSV results saved to: {csv_file_path}")

        if any(r and "Error" not in r for r in raw_results_for_plot): plot_results(raw_results_for_plot, plot_file)
        else: print("No successful benchmark results available for plotting.")

    except KeyboardInterrupt: print("\nBenchmark interrupted by user."); exit_code = 130
    except SystemExit as e:
        exit_code = e.code if e.code is not None else 0
        if exit_code != 0 : print(f"Script exited via SystemExit (Code: {exit_code}).")
    except Exception as e:
        print(f"\n[Error] Unexpected error in main: {e}")
        import traceback; traceback.print_exc()
        exit_code = 1
    finally:
        if pm: pm.cleanup_at_exit()
        print(f"Script execution finished. Final exit code: {exit_code}")
        current_exception = sys.exc_info()[1] 
        if exit_code != 0 and not isinstance(current_exception, SystemExit):
            sys.exit(exit_code)

if __name__ == "__main__":
    main()