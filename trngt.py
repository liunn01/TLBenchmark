import argparse
from inference_benchmark import run_inference_benchmark, add_inference_args
from gemm_benchmark import run_gemm_benchmark, add_gemm_args
from nccl_benchmark import run_nccl_benchmark, add_nccl_args
from baseinfo import run_baseinfo, add_baseinfo_args
import sys
import os
import pandas as pd
from datetime import datetime
import re
import subprocess
import copy

def build_parser(selected_types):
    parser = argparse.ArgumentParser(
        description='vLLM/GEMM/NCCL/Baseinfo Benchmark Entry',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # General arguments
    parser.add_argument('--inference', action='store_true', help='Run vLLM inference benchmark.')
    parser.add_argument('--gemm', action='store_true', help='Run GEMM test.')
    parser.add_argument('--nccl', action='store_true', help='Run NCCL test.')
    parser.add_argument('--baseinfo', action='store_true', help='Run Baseinfo test.')
    parser.add_argument('--log-dir', type=str, default="./benchmark_logs", help='Directory for all logs and results.')
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit.')

    if 'inference' in selected_types:
        add_inference_args(parser)
    if 'gemm' in selected_types:
        add_gemm_args(parser)
    if 'nccl' in selected_types:
        add_nccl_args(parser)
    if 'baseinfo' in selected_types:
        add_baseinfo_args(parser)
    return parser

def parse_arguments():
    # First, parse the main function type
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--inference', action='store_true')
    base_parser.add_argument('--gemm', action='store_true')
    base_parser.add_argument('--nccl', action='store_true')
    base_parser.add_argument('--baseinfo', action='store_true')
    base_parser.add_argument('--log-dir', type=str, default="./benchmark_logs_client_only")
    base_parser.add_argument('-h', '--help', action='store_true')
    known, _ = base_parser.parse_known_args()
    
    selected_types = []
    if known.inference: selected_types.append('inference')
    if known.gemm: selected_types.append('gemm')
    if known.nccl: selected_types.append('nccl')
    if known.baseinfo: selected_types.append('baseinfo')
    
    # If nothing is selected, select all
    if not selected_types and not known.help:
        selected_types = ['inference', 'gemm', 'nccl', 'baseinfo']
    
    parser = build_parser(selected_types)
    
    # When help is the only argument, or a primary flag is passed with help, show help and exit
    is_primary_flag_with_help = len(sys.argv) == 3 and any(x in sys.argv for x in ['--inference','--gemm','--nccl','--baseinfo'])
    if known.help and (len(sys.argv) == 2 or is_primary_flag_with_help):
        parser.print_help()
        sys.exit(0)
        
    # Also compatible with cases like: python main.py --inference --help
    if known.help:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    return args

def parse_sweep_args(args):
    """
    Parses multiple sets of inference parameters, supporting the following format:
    --random-input-len "100,200" --random-output-len "1000,2000" --concurrency-levels "[2,4,8],[2,4]"
    Returns a list of parameter groups, each as a dict.
    """
    input_lens = [x.strip() for x in str(getattr(args, 'random_input_len', '')).split(',')]
    output_lens = [x.strip() for x in str(getattr(args, 'random_output_len', '')).split(',')]
    
    # Extract content from each []
    conc_str = str(getattr(args, 'concurrency_levels', ''))
    conc_groups = re.findall(r'\[([^\]]+)\]', conc_str)
    concurrency_levels = [g.strip() for g in conc_groups]
    
    # If no [], split by comma
    if not concurrency_levels:
        concurrency_levels = [conc_str.strip()] * max(len(input_lens), len(output_lens))
        
    # Align lengths
    n = min(len(input_lens), len(output_lens), len(concurrency_levels))
    param_groups = []
    for i in range(n):
        param_groups.append({
            'random_input_len': input_lens[i],
            'random_output_len': output_lens[i],
            'concurrency_levels': concurrency_levels[i],
        })
    return param_groups

def kill_gpu_processes():
    """
    Finds and kills all processes occupying the GPU (via nvidia-smi).
    """
    try:
        # Get PIDs of all processes using GPU
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(f"Nvidia-smi error: {result.stderr}")
            return
        pids = set()
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        if not pids:
            print("No GPU processes found to kill.")
            return
        print(f"Killing GPU processes: {pids}")
        for pid in pids:
            try:
                os.kill(pid, 9) # SIGKILL
                print(f"Killed process {pid}")
            except Exception as e:
                print(f"Failed to kill process {pid}: {e}")
    except FileNotFoundError:
        print("Error: `nvidia-smi` command not found. Is NVIDIA driver installed?")
    except Exception as e:
        print(f"An error occurred while trying to kill GPU processes: {e}")

def merge_results_to_summary(log_dir, inference_data=None, gemm_path=None, nccl_data=None, baseinfo_data=None):
    summary_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(log_dir, f"summary_{summary_time}.xlsx")
    wrote_any_sheet = False
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        # Inference
        if inference_data is not None:
            if isinstance(inference_data, pd.DataFrame):
                df = inference_data
            elif isinstance(inference_data, str) and os.path.exists(inference_data):
                df = pd.read_csv(inference_data) if inference_data.endswith('.csv') else pd.read_excel(inference_data)
            else:
                df = None
            if df is not None:
                df.to_excel(writer, sheet_name='inference', index=False)
                wrote_any_sheet = True
        # GEMM
        if gemm_path and os.path.exists(gemm_path):
            gemm_cols = ["GPU_ID", "Operation", "m", "n", "k", "GB/s", "GFLOPs"]
            try:
                df = pd.read_csv(gemm_path, usecols=gemm_cols) if gemm_path.endswith('.csv') else pd.read_excel(gemm_path, usecols=gemm_cols)
                df.to_excel(writer, sheet_name='gemm', index=False)
                wrote_any_sheet = True
            except Exception as e:
                print(f"Failed to read GEMM results from {gemm_path}: {e}")
        # NCCL
        if nccl_data is not None:
            try:
                if isinstance(nccl_data, dict):
                    rows = []
                    # CUDA Bandwidth Test
                    bw = nccl_data.get('bandwidth', {})
                    if bw:
                        rows.append({'Section': 'CUDA Bandwidth Test', 'Test Item': '', 'Value': ''})
                        rows.append({'Section': '', 'Test Item': 'Host to Device (H2D) Avg/Card', 'Value': f"{bw.get('h2d', 'N/A'):.2f} GB/s" if 'h2d' in bw else 'N/A'})
                        rows.append({'Section': '', 'Test Item': 'Device to Host (D2H) Avg/Card', 'Value': f"{bw.get('d2h', 'N/A'):.2f} GB/s" if 'd2h' in bw else 'N/A'})
                        rows.append({'Section': '', 'Test Item': 'Device to Device (D2D) Avg/Card', 'Value': f"{bw.get('d2d', 'N/A'):.2f} GB/s" if 'd2d' in bw else 'N/A'})
                    # P2P Bandwidth Test
                    p2p = nccl_data.get('p2p', {})
                    if p2p:
                        rows.append({'Section': 'P2P Bandwidth Test', 'Test Item': '', 'Value': ''})
                        rows.append({'Section': '', 'Test Item': 'Unidirectional P2P Bandwidth', 'Value': f"{p2p.get('unidirectional_avg', 'N/A'):.2f} GB/s" if 'unidirectional_avg' in p2p else 'N/A'})
                        rows.append({'Section': '', 'Test Item': 'Bidirectional P2P Bandwidth', 'Value': f"{p2p.get('bidirectional_avg', 'N/A'):.2f} GB/s" if 'bidirectional_avg' in p2p else 'N/A'})
                    # NCCL Tests
                    nccl_keys = [k for k in nccl_data if k.startswith('nccl_')]
                    if nccl_keys:
                        rows.append({'Section': 'NCCL Tests', 'Test Item': '', 'Value': ''})
                        for k in nccl_keys:
                            v = nccl_data[k]
                            test_type = v.get('test_type', k.replace('nccl_', '').replace('_', ' ').title())
                            avg_bw = v.get('avg_bus_bw', 'N/A')
                            value = f"{avg_bw:.2f} GB/s" if isinstance(avg_bw, float) or (isinstance(avg_bw, str) and avg_bw.replace('.', '', 1).isdigit()) else avg_bw
                            rows.append({'Section': '', 'Test Item': test_type, 'Value': value})
                    df = pd.DataFrame(rows)
                    df.to_excel(writer, sheet_name='nccl', index=False)
                    wrote_any_sheet = True
                elif isinstance(nccl_data, str) and os.path.exists(nccl_data):
                    nccl_xls = pd.ExcelFile(nccl_data)
                    for sheet in nccl_xls.sheet_names:
                        df = nccl_xls.parse(sheet)
                        df.to_excel(writer, sheet_name=sheet, index=False)
                        wrote_any_sheet = True
            except Exception as e:
                print(f"Failed to write NCCL summary: {e}")
        # Baseinfo
        if baseinfo_data is not None:
            if isinstance(baseinfo_data, pd.DataFrame):
                df = baseinfo_data
            elif isinstance(baseinfo_data, str) and os.path.exists(baseinfo_data):
                df = pd.read_csv(baseinfo_data) if baseinfo_data.endswith('.csv') else pd.read_excel(baseinfo_data)
            else:
                df = None
            if df is not None:
                df.to_excel(writer, sheet_name='baseinfo', index=False)
                wrote_any_sheet = True

    if wrote_any_sheet:
        print(f"Summary results have been saved to: {summary_path}")
    else:
        if os.path.exists(summary_path):
            os.remove(summary_path)
        print("No summarizable result files were generated this time, summary.xlsx was not created.")

def get_new_files(before_set, after_set):
    """Returns a list of new files (csv/xlsx) in 'after_set' that were not in 'before_set'."""
    return [f for f in after_set if f not in before_set and (f.endswith('.csv') or f.endswith('.xlsx'))]

def main():
    args = parse_arguments()
    orig_log_dir = getattr(args, 'log_dir', './benchmark_logs_client_only')
    
    if getattr(args, 'model', None):
        args.inferred_log_model_name = args.model.replace("/", "_").replace("\\", "_")
        
    ran_any = False
    inference_result_paths = []
    gemm_result_path = None
    nccl_result_path = None
    baseinfo_df = None
    nccl_summary_data = None

    inf_dir = os.path.join(orig_log_dir, 'inference')
    gemm_dir = os.path.join(orig_log_dir, 'gemm')
    nccl_dir = os.path.join(orig_log_dir, 'nccl')
    baseinfo_dir = os.path.join(orig_log_dir, 'baseinfo')

    inf_files_before = set(os.listdir(inf_dir)) if os.path.exists(inf_dir) else set()
    gemm_files_before = set(os.listdir(gemm_dir)) if os.path.exists(gemm_dir) else set()
    nccl_files_before = set(os.listdir(nccl_dir)) if os.path.exists(nccl_dir) else set()
    baseinfo_files_before = set(os.listdir(baseinfo_dir)) if os.path.exists(baseinfo_dir) else set()

    # 1. Baseinfo
    if getattr(args, 'baseinfo', False):
        print("\n--- Running Baseinfo Test ---")
        args.log_dir = baseinfo_dir
        baseinfo_df = run_baseinfo(args)  # Get DataFrame directly
        ran_any = True
        print("--- Baseinfo Test Finished ---\n")

    # 2. Inference
    inference_merged_df = None
    ran_inference = False
    if getattr(args, 'inference', False):
        print("\n--- Running Inference Test ---")
        args.log_dir = inf_dir
        param_groups = parse_sweep_args(args)
        if len(param_groups) > 1:
            print(f"\nDetected {len(param_groups)} sets of inference parameters, executing sequentially...\n")
            
        for idx, param in enumerate(param_groups):
            if len(param_groups) > 1:
                print(f"\n===== Starting inference test group {idx+1}/{len(param_groups)}: input_len={param['random_input_len']}, output_len={param['random_output_len']}, concurrency_levels={param['concurrency_levels']} =====")
            
            args_one = copy.deepcopy(args)
            args_one.random_input_len = int(param['random_input_len'])
            args_one.random_output_len = int(param['random_output_len'])
            args_one.concurrency_levels = param['concurrency_levels']
            run_inference_benchmark(args_one)
            
            inf_files_after = set(os.listdir(inf_dir)) if os.path.exists(inf_dir) else set()
            new_inf_files = get_new_files(inf_files_before, inf_files_after)
            if new_inf_files:
                new_inf_files.sort(key=lambda x: os.path.getmtime(os.path.join(inf_dir, x)), reverse=True)
                new_csv_files = [f for f in new_inf_files if f.endswith('.csv')]
                if new_csv_files:
                    inference_result_paths.append(os.path.join(inf_dir, new_csv_files[0]))
                inf_files_before = inf_files_after.copy()
            if len(param_groups) > 1:
                 print(f"===== Inference test group {idx+1}/{len(param_groups)} finished =====\n")
        ran_any = True
        ran_inference = True
        print("--- Inference Test Finished ---\n")

    if inference_result_paths:
        all_dfs = []
        for p in inference_result_paths:
            try:
                df = pd.read_csv(p)
                all_dfs.append(df)
            except Exception as e:
                print(f"Failed to read CSV {p}: {e}")
        if all_dfs:
            inference_merged_df = pd.concat(all_dfs, ignore_index=True)
            # 输出一次终端Final Benchmark Summary表格
            try:
                from inference_benchmark import print_formatted_table
                # 构造表格数据
                if not inference_merged_df.empty:
                    header = list(inference_merged_df.columns)
                    table_data = [header] + inference_merged_df.astype(str).values.tolist()
                    print("\n" + "="*25 + " Final Benchmark Summary " + "="*25)
                    print_formatted_table(table_data)
                    print("="*73 + "\n")
            except Exception as e:
                print(f"[Warning] Could not print formatted summary table: {e}")

    run_gemm = getattr(args, 'gemm', False)
    run_nccl = getattr(args, 'nccl', False)
    if ran_inference and (run_gemm or run_nccl):
        print("[INFO] GEMM or NCCL test will run after inference. Killing existing GPU processes to ensure a clean state...")
        kill_gpu_processes()

    # 3. GEMM
    if run_gemm:
        print("\n--- Running GEMM Test ---")
        args.log_dir = gemm_dir
        os.makedirs(gemm_dir, exist_ok=True)
        run_gemm_benchmark(args)
        ran_any = True
        gemm_files_after = set(os.listdir(gemm_dir)) if os.path.exists(gemm_dir) else set()
        new_gemm_files = get_new_files(gemm_files_before, gemm_files_after)
        if new_gemm_files:
            new_gemm_files.sort(key=lambda x: os.path.getmtime(os.path.join(gemm_dir, x)), reverse=True)
            gemm_result_path = os.path.join(gemm_dir, new_gemm_files[0])
        print("--- GEMM Test Finished ---\n")

    # 4. NCCL
    if run_nccl:
        print("\n--- Running NCCL Test ---")
        args.log_dir = nccl_dir
        nccl_summary_data = run_nccl_benchmark(args)
        ran_any = True
        nccl_files_after = set(os.listdir(nccl_dir)) if os.path.exists(nccl_dir) else set()
        new_nccl_files = get_new_files(nccl_files_before, nccl_files_after)
        if new_nccl_files:
            new_nccl_files.sort(key=lambda x: os.path.getmtime(os.path.join(nccl_dir, x)), reverse=True)
            nccl_result_path = os.path.join(nccl_dir, new_nccl_files[0])
        print("--- NCCL Test Finished ---\n")

    if not ran_any:
        print("No test type specified. Please use --inference, --gemm, --nccl, or --baseinfo.")
        sys.exit(1)

    # Summarize all newly generated results from this run
    print("\n--- Merging all results into a summary file ---")
    merge_results_to_summary(orig_log_dir, inference_merged_df, gemm_result_path, nccl_summary_data, baseinfo_df)

if __name__ == "__main__":
    main()
