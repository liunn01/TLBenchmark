import os
import sys
import time
import signal
import subprocess
import requests
import threading
import argparse
from typing import List, Optional
from datetime import datetime
import psutil
import random
import csv

# Force the use of a non-graphical backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='GPU Performance Benchmark Tool')
    
    # Server configuration
    parser.add_argument('--model', type=str, default="/local/models/DeepSeek-R1-Distill-Llama-8B",
                        help='Path to the model (default: %(default)s)')
    parser.add_argument('--host', type=str, default="0.0.0.0",  # 修改默认值为 0.0.0.0
                        help='Host to run the server on (default: %(default)s)')
    parser.add_argument('--port', type=int, default=8335,
                        help='Port to run the server on (default: %(default)s)')
    parser.add_argument('--gpu', type=str, default="0",
                        help='GPU device IDs to use (comma-separated) (default: %(default)s)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Tensor parallelism size (default: %(default)s)')
    parser.add_argument('--data-parallel-size', type=int, default=1,
                        help='Data parallelism size (default: %(default)s)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=131072,  # 修改默认值为 131072
                        help='Maximum number of batched tokens (default: %(default)s)')
    parser.add_argument('--max_num_seqs', type=int, default=256,
                        help='Maximum number of sequences in a batch (default: %(default)s)')
    
    # Benchmark configuration
    parser.add_argument('--random-input-len', type=int, default=20,
                        help='Length of random input tokens (default: %(default)s)')
    parser.add_argument('--random-output-len', type=int, default=20,
                        help='Length of random output tokens (default: %(default)s)')
    parser.add_argument('--concurrency-levels', type=str, default="4,8,16,32,64,128,256",
                        help='Concurrency levels to test (comma-separated) (default: %(default)s)')
    parser.add_argument('--prompts-multiplier', type=int, default=5,
                        help='Multiplier for number of prompts relative to concurrency (default: %(default)s)')
    parser.add_argument('--log-dir', type=str, default="./benchmark_logs",
                        help='Directory for logs (default: %(default)s)')
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='Trust remote code when loading models (default: %(default)s)')
    
    return parser.parse_args()

class ProcessManager:
    """Process management class, responsible for starting, monitoring, and cleaning up processes"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.log_threads: List[threading.Thread] = []
        self._register_signals()
    
    def _register_signals(self):
        """Register signal handlers"""
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
    
    def start_process(self, cmd: List[str], log_file: str, prefix: str = "") -> subprocess.Popen:
        """Start a process and set up log redirection"""
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Start the process
        proc = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.processes.append(proc)
        print(f"Started process: PID={proc.pid}, CMD={' '.join(cmd)}")
        
        # Start log recording thread
        def log_worker():
            try:
                with open(log_file, "a") as f:  # Use append mode
                    while proc.poll() is None:  # Check if the process has exited
                        line = proc.stdout.readline()
                        if line:
                            output = f"[{prefix}] {line.strip()}"
                            print(output)
                            f.write(f"{datetime.now().isoformat()} {output}\n")
            except Exception as e:
                print(f"Error in log worker: {e}")
        
        log_thread = threading.Thread(target=log_worker, daemon=True)
        log_thread.start()
        self.log_threads.append(log_thread)
        
        return proc
    
    def cleanup(self, signum=None, frame=None):
        """Clean up all processes"""
        print("\n[Cleanup] Terminating processes...")
        
        # Track cleanup success
        cleanup_success = True
        
        # Terminate child processes
        for proc in self.processes:
            try:
                if not proc or proc.poll() is not None:
                    continue  # Skip if process is not running
                    
                print(f"Terminating process: PID={proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print(f"Process {proc.pid} terminated successfully")
                except subprocess.TimeoutExpired:
                    print(f"Process {proc.pid} did not terminate, sending SIGKILL")
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                        print(f"Process {proc.pid} killed successfully")
                    except subprocess.TimeoutExpired:
                        print(f"Failed to kill process {proc.pid}")
                        cleanup_success = False
            except (ProcessLookupError, AttributeError) as e:
                print(f"Process already gone: {e}")
            except Exception as e:
                print(f"Error terminating process: {e}")
                cleanup_success = False
            finally:
                if proc and proc.stdout:
                    proc.stdout.close()  # Ensure resources are released
        
        # Clean up remaining processes with more robust methods
        try:
            # Force terminate child processes
            for proc in self.processes:
                try:
                    if proc and proc.pid and psutil.pid_exists(proc.pid):
                        parent = psutil.Process(proc.pid)
                        children = parent.children(recursive=True)  # Get all child processes
                        for child in children:
                            try:
                                print(f"Force killing child process: PID={child.pid}")
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                                print(f"Error killing child process {child.pid}: {e}")
                except (psutil.NoSuchProcess, AttributeError):
                    pass
                except Exception as e:
                    print(f"Error in child process cleanup: {e}")
                    cleanup_success = False
        except Exception as e:
            print(f"Error cleaning up child processes: {e}")
            cleanup_success = False
            
        try:
            self._kill_zombie_processes()
        except Exception as e:
            print(f"Error in zombie process cleanup: {e}")
            cleanup_success = False
            
        try:
            self._kill_process_by_port(args.port)  # Kill process by port
        except Exception as e:
            print(f"Error in port process cleanup: {e}")
            cleanup_success = False
            
        try:
            self._kill_vllm_main_process(args.port)  # Kill vLLM main process by port
        except Exception as e:
            print(f"Error in vLLM main process cleanup: {e}")
            cleanup_success = False
            
        try:
            self._kill_gpu_processes()
        except Exception as e:
            print(f"Error in GPU process cleanup: {e}")
            cleanup_success = False
            
        print("[Cleanup] " + ("Completed successfully" if cleanup_success else "Completed with errors"))
        
        # Ensure program exit
        sys.exit(0)
    
    def _kill_zombie_processes(self):
        """Clean up remaining Python processes"""
        zombie_count = 0
        error_count = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    if 'cmdline' in proc.info and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        if 'vllm' in cmdline or 'benchmark_serving' in cmdline:
                            print(f"Force killing zombie process: PID={proc.info['pid']}, CMD={cmdline}")
                            proc.kill()
                            zombie_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as e:
                print(f"Error killing zombie process: {e}")
                error_count += 1
                
        print(f"Zombie process cleanup: {zombie_count} processes killed, {error_count} errors")
    
    def _kill_process_by_port(self, port: int):
        """Find and terminate processes by port"""
        found = False
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    pid = conn.pid
                    if pid:
                        try:
                            proc = psutil.Process(pid)
                            print(f"Killing process listening on port {port}: PID={pid}, Name={proc.name()}")
                            proc.kill()
                            found = True
                            print(f"Successfully killed process on port {port}")
                        except psutil.NoSuchProcess:
                            print(f"Process {pid} on port {port} no longer exists")
                        except Exception as e:
                            print(f"Error killing process {pid} on port {port}: {e}")
            
            if not found:
                print(f"No process found listening on port {port}")
                
        except Exception as e:
            print(f"Failed to search for processes on port {port}: {e}")
    
    def _kill_vllm_main_process(self, port: int):
        """Find and kill vLLM serve main process by port"""
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                    pid = conn.pid
                    if pid:
                        try:
                            proc = psutil.Process(pid)
                            print(f"Killing vllm serve main process: PID={pid}, Name={proc.name()}")
                            
                            # Get children before killing parent
                            children = []
                            try:
                                children = proc.children(recursive=True)
                            except Exception as e:
                                print(f"Error getting children of process {pid}: {e}")
                            
                            # Terminate child processes recursively
                            for child in children:
                                try:
                                    print(f"Killing child process: PID={child.pid}, Name={child.name()}")
                                    child.kill()
                                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                                    print(f"Error killing child {child.pid}: {e}")
                                    
                            # Kill parent process
                            proc.kill()
                            print(f"Successfully killed vLLM main process PID={pid}")
                            return
                        except psutil.NoSuchProcess:
                            print(f"vLLM process {pid} no longer exists")
                        except Exception as e:
                            print(f"Error killing vLLM process {pid}: {e}")
            
            print("No vLLM main process found on specified port")
        except Exception as e:
            print(f"Failed to search for vLLM main process on port {e}")
    
    def _kill_gpu_processes(self):
        """Find and terminate processes using the GPU through nvidia-smi"""
        try:
            # Use nvidia-smi to find processes using GPUs
            output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            pids = output.decode().strip().split('\n')
            
            if not pids or (len(pids) == 1 and not pids[0].strip()):
                print("No GPU processes found")
                return
                
            killed_count = 0
            error_count = 0
            
            for pid in pids:
                if pid.strip():
                    try:
                        pid = int(pid.strip())
                        proc = psutil.Process(pid)
                        proc_name = proc.name()
                        print(f"Force killing GPU process: PID={pid}, Name={proc_name}")
                        proc.kill()
                        killed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"Cannot kill GPU process {pid}: {e}")
                        error_count += 1
                    except ValueError as e:
                        print(f"Invalid PID format: {pid}")
                        error_count += 1
                    except Exception as e:
                        print(f"Error killing GPU process {pid}: {e}")
                        error_count += 1
            
            print(f"GPU process cleanup: {killed_count} processes killed, {error_count} errors")
        except subprocess.CalledProcessError as e:
            print(f"nvidia-smi command failed: {e}")
        except FileNotFoundError:
            print("nvidia-smi command not found. NVIDIA drivers may not be installed.")
        except Exception as e:
            print(f"Failed to kill GPU processes: {e}")

def wait_for_server(host: str, port: int, timeout: int = 600) -> bool:
    """Wait for server to start"""
    print(f"Waiting for server on {host}:{port}...")
    start_time = time.time()
    last_print = 0
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"http://{host}:{port}/v1/completions",
                timeout=5
            )
            if response.status_code == 405:
                print("\nServer is ready!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            if time.time() - last_print > 30:
                print(". ", end="", flush=True)
                last_print = time.time()
            time.sleep(1)
        except Exception as e:
            print(f"\nError checking server status: {str(e)}")
            break
    return False

def run_benchmark(pm: ProcessManager, config: dict, log_file: str):
    """Execute benchmark test"""
    cmd = [
        "vllm bench serve",
        "--dataset-name", "random",
        "--random-input-len", str(config['input_len']),
        "--random-output-len", str(config['output_len']),
        "--model", config['model'],
        "--host", config['host'],
        "--port", str(config['port']),
        "--num-prompts", str(config['num_prompts']),
        "--seed", str(config['seed']),
        "--max-concurrency", str(config['concurrency'])
    ]
    
    # Define regex patterns for lines to keep
    grep_patterns = [
        "Maximum request concurrency:",
        r"Successful requests:\s+[0-9]+",
        r"Benchmark duration \(s\):\s+[0-9.]+",
        r"Request throughput \(req/s\):\s+[0-9.]+",
        r"Output token throughput \(tok/s\):\s+[0-9.]+",
        r"Total Token throughput \(tok/s\):\s+[0-9.]+",
        r"Median TTFT \(ms\):\s+[0-9.]+",
        r"Median TPOT \(ms\):\s+[0-9.]+"
    ]
    
    # Construct filter command
    grep_cmd = " | ".join([
        " ".join(cmd),
        "grep -E --color=never '" + "|".join(grep_patterns) + "'"
    ])
    
    print(f"Running benchmark with concurrency={config['concurrency']}")
    print(f"Command: {grep_cmd}")
    
    try:
        # Execute the filtered command directly with subprocess
        proc = subprocess.Popen(
            grep_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            error_msg = f"Benchmark failed with exit code {proc.returncode}"
            if stderr:
                error_msg += f": {stderr}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        if not stdout.strip():
            print("Warning: Benchmark returned empty results")
            return {"MaxConcurrency": str(config['concurrency']), "Error": "Empty results"}
        
        # Parse and format output
        results = parse_benchmark_output(stdout)
        
        # Add concurrency level if not in results
        if "MaxConcurrency" not in results:
            results["MaxConcurrency"] = str(config['concurrency'])
            
        return results  # Return parsed results
    except subprocess.SubprocessError as e:
        print(f"Subprocess error during benchmark: {e}")
        return {"MaxConcurrency": str(config['concurrency']), "Error": str(e)}
    except Exception as e:
        print(f"Error during benchmark with concurrency={config['concurrency']}: {e}")
        return {"MaxConcurrency": str(config['concurrency']), "Error": str(e)}

def parse_benchmark_output(output: str) -> dict:
    """Parse benchmark test output"""
    results = {}
    for line in output.splitlines():
        if "Maximum request concurrency:" in line:
            results["MaxConcurrency"] = line.split(":")[1].strip()
        elif "Successful requests:" in line:
            results["SuccessRequests"] = line.split(":")[1].strip()
        elif "Request throughput (req/s):" in line:
            results["RequestThroughput"] = line.split(":")[1].strip()
        elif "Output token throughput" in line:
            results["OutputTokenThroughput"] = line.split(":")[1].strip()
        elif "Total Token throughput" in line:
            results["TotalTokenThroughput"] = line.split(":")[1].strip()
        elif "Median TTFT" in line:
            results["TTFT"] = line.split(":")[1].strip()
        elif "Median TPOT" in line:
            results["TPOT"] = line.split(":")[1].strip()
    return results

def format_benchmark_results(results: dict) -> str:
    """Format benchmark results"""
    header = format_benchmark_header()
    row = format_benchmark_row(results)
    return header + "\n" + row

def format_benchmark_header() -> str:
    """Generate title row for benchmark results"""
    return "{:<10} {:<10} {:<10} {:<15} {:<15} {:<10} {:<10}".format(
        "Max-Conc",  # Maximum-request-concurrency
        "Succ-Req",  # Successful-requests
        "Req/s",     # Request-throughput(req/s)
        "Out-Tok/s", # Output-token-throughput(tok/s)
        "Tot-Tok/s", # Total-Token-throughput(tok/s)
        "TTFT",      # Median-TTFT(ms)
        "TPOT"       # Median-TPOT(ms)
    )

def format_benchmark_row(results: dict) -> str:
    """Generate data row for benchmark results"""
    return "{:<10} {:<10} {:<10} {:<15} {:<15} {:<10} {:<10}".format(
        results.get("MaxConcurrency", "N/A"),
        results.get("SuccessRequests", "N/A"),
        results.get("RequestThroughput", "N/A"),
        results.get("OutputTokenThroughput", "N/A"),
        results.get("TotalTokenThroughput", "N/A"),
        results.get("TTFT", "N/A"),
        results.get("TPOT", "N/A")
    )

def print_formatted_results(results: dict):
    """Format and print benchmark results"""
    header = [
        "MaxConcurrency",
        "SuccessRequests",
        "OutputTokenThroughput",
        "TotalTokenThroughput",
        "TTFT",
        "TPOT"
    ]
    print("\t".join(header))  # Print the header row
    row = [
        results.get("MaxConcurrency", "N/A"),
        results.get("SuccessRequests", "N/A"),
        results.get("OutputTokenThroughput", "N/A"),
        results.get("TotalTokenThroughput", "N/A"),
        results.get("TTFT", "N/A"),
        results.get("TPOT", "N/A")
    ]
    print("\t".join(row))  # Print the data row

def plot_results(results_list, output_file="benchmark_results.png"):
    """Generate chart based on benchmark results"""
    # Extract data
    max_conc = [int(result["MaxConcurrency"]) for result in results_list if "MaxConcurrency" in result and "Error" not in result]
    
    # Skip if no valid results
    if not max_conc:
        print("No valid benchmark results to plot")
        return
        
    req_per_sec = [float(result["RequestThroughput"]) for result in results_list if "RequestThroughput" in result]
    out_tok_per_sec = [float(result["OutputTokenThroughput"]) for result in results_list if "OutputTokenThroughput" in result]
    tot_tok_per_sec = [float(result["TotalTokenThroughput"]) for result in results_list if "TotalTokenThroughput" in result]
    ttft = [float(result["TTFT"]) for result in results_list if "TTFT" in result]
    tpot = [float(result["TPOT"]) for result in results_list if "TPOT" in result]
    
    # Set chart size
    plt.figure(figsize=(10, 6))
    
    # Plot line graphs for each field
    if req_per_sec: plt.plot(max_conc[:len(req_per_sec)], req_per_sec, label="Req/s", marker="o")
    if out_tok_per_sec: plt.plot(max_conc[:len(out_tok_per_sec)], out_tok_per_sec, label="Out-Tok/s", marker="o")
    if tot_tok_per_sec: plt.plot(max_conc[:len(tot_tok_per_sec)], tot_tok_per_sec, label="Tot-Tok/s", marker="o")
    if ttft: plt.plot(max_conc[:len(ttft)], ttft, label="TTFT", marker="o")
    if tpot: plt.plot(max_conc[:len(tpot)], tpot, label="TPOT", marker="o")
    
    # Set title and labels
    plt.title("Benchmark Results", fontsize=16)
    plt.xlabel("Max-Conc", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)
    
    # Set y-axis ticks at intervals of 200 if we have data
    if any([req_per_sec, out_tok_per_sec, tot_tok_per_sec, ttft, tpot]):
        max_val = max([
            max(req_per_sec) if req_per_sec else 0,
            max(out_tok_per_sec) if out_tok_per_sec else 0,
            max(tot_tok_per_sec) if tot_tok_per_sec else 0,
            max(ttft) if ttft else 0,
            max(tpot) if tpot else 0
        ])
        plt.yticks(range(0, int(max_val + 200), 200))
    
    # Display legend
    plt.legend()
    
    # Display grid
    plt.grid(True, linestyle="-", alpha=0.6)
    
    # Save chart to file
    try:
        plt.savefig(output_file)
        print(f"Saved benchmark results plot to {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Don't include plt.show() to avoid attempting to display in a command-line environment

def main():
    global args
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create log directories
        log_dirs = [
            os.path.join(args.log_dir, "vllm_server_log"),
            os.path.join(args.log_dir, "benchmark_output_log")
        ]
        os.makedirs(log_dirs[0], exist_ok=True)
        os.makedirs(log_dirs[1], exist_ok=True)
        
        # Initialize process manager
        pm = ProcessManager()
        
        # Prepare log files
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        server_log = os.path.join(log_dirs[0], f"vllm-{current_time}.log")
        benchmark_log = os.path.join(log_dirs[1], f"benchmark-{current_time}.log")
        csv_file = os.path.join(log_dirs[1], f"benchmark-{current_time}.csv")  # CSV file path
        plot_file = os.path.join(log_dirs[1], f"benchmark-{current_time}.png")  # Chart file path
        
        # Start vLLM server
        vllm_cmd = [
            f"CUDA_VISIBLE_DEVICES={args.gpu}",
            "vllm", "serve", args.model,
            "--host", args.host,
            "--port", str(args.port),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--data-parallel-size", str(args.data_parallel_size),  # 新增参数
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--max-num-seqs", str(args.max_num_seqs)
        ]
        
        # Add trust-remote-code flag if specified
        if args.trust_remote_code:
            vllm_cmd.append("--trust-remote-code")
        
        pm.start_process(vllm_cmd, server_log, "vLLM")
        
        # Wait for server to start
        if not wait_for_server(args.host, args.port, timeout=1200):
            raise RuntimeError("Server failed to start within timeout")
        
        # Write CSV file header
        with open(csv_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "Max-Conc",  # Maximum-request-concurrency
                "Succ-Req",  # Successful-requests
                "Req/s",     # Request-throughput(req/s)
                "Out-Tok/s", # Output-token-throughput(tok/s)
                "Tot-Tok/s", # Total-Token-throughput(tok/s)
                "TTFT",      # Median-TTFT(ms)
                "TPOT"       # Median-TPOT(ms)
            ])
        
        # Write log file header
        with open(benchmark_log, "w") as f:
            f.write(format_benchmark_header() + "\n")
        
        # Run benchmark tests
        config = {
            "model": args.model,
            "host": args.host,
            "port": args.port,
            "input_len": args.random_input_len,
            "output_len": args.random_output_len,
            "num_prompts": 0,
            "concurrency": None
        }
        
        results_list = []  # Store all benchmark results
        
        # Parse concurrency levels
        concurrency_levels = [int(c) for c in args.concurrency_levels.split(',')]
        
        for concurrency in concurrency_levels:
            print(f"\n{'='*40}")
            print(f"Starting concurrency test: {concurrency}")
            print(f"{'='*40}")
            
            # Generate a random seed
            random_seed = random.randint(10000, 99999)
            
            # Update configuration
            config['concurrency'] = concurrency
            config['num_prompts'] = concurrency * args.prompts_multiplier  # Set num-prompts relative to concurrency
            config['seed'] = random_seed  # Add random seed to configuration
            
            print(f"Using random seed: {random_seed}")
            print(f"Setting num-prompts to {config['num_prompts']} ({args.prompts_multiplier}x max-concurrency)")
            
            # Execute benchmark test
            results = run_benchmark(pm, config, benchmark_log)
            results_list.append(results)  # Collect results
            
            # Skip logging if benchmark failed
            if "Error" in results:
                print(f"Skipping logging for failed benchmark with concurrency={concurrency}: {results['Error']}")
                continue
                
            # Write to log file
            with open(benchmark_log, "a") as f:
                f.write(format_benchmark_row(results) + "\n")
            
            # Write to CSV file
            with open(csv_file, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    results.get("MaxConcurrency", "N/A"),
                    results.get("SuccessRequests", "N/A"),
                    results.get("RequestThroughput", "N/A"),
                    results.get("OutputTokenThroughput", "N/A"),
                    results.get("TotalTokenThroughput", "N/A"),
                    results.get("TTFT", "N/A"),
                    results.get("TPOT", "N/A")
                ])
        
        # Generate chart after benchmark tests if we have results
        if any(["Error" not in result for result in results_list]):
            plot_results(results_list, plot_file)
        else:
            print("No successful benchmark results to plot")
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        if 'pm' in locals():
            pm.cleanup()
        sys.exit(0)  # 确保程序退出
    except Exception as e:
        print(f"\n[Error] {str(e)}")
        sys.exit(1)
    finally:
        try:
            if 'pm' in locals():
                pm.cleanup()
        except Exception as e:
            print(f"Error during final cleanup: {e}")
    
    # Check for remaining Python processes
    try:
        print("\n[Debug] Checking for remaining Python processes...")
        remaining_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    if 'cmdline' in proc.info and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        print(f"Remaining process: PID={proc.info['pid']}, CMD={cmdline}")
                        remaining_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if remaining_count == 0:
            print("No remaining Python processes found.")
    except Exception as e:
        print(f"Error checking remaining processes: {e}")

if __name__ == "__main__":
    main()