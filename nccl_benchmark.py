import subprocess
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import datetime
import os
import re
import sys
import argparse

# --- Logger Class ---
class Logger:
    """A class to write print output to both a file and the terminal."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if self.log:
            self.log.close()

# --- Parsing Functions ---
def parse_bandwidth_test_output(output: str) -> dict:
    """Parses the output of the bandwidthTest executable."""
    results = {}
    try:
        devices = re.findall(r"Device\s\d+:", output)
        num_devices = len(devices) if len(devices) > 0 else 1
        patterns = {
            "h2d": r"Host to Device Bandwidth,.*?\n\s*PINNED Memory Transfers\n\s*Transfer Size \(Bytes\)\s*Bandwidth\(GB/s\)\n((?:\s*\d+\s+[\d\.]+\n)+)",
            "d2h": r"Device to Host Bandwidth,.*?\n\s*PINNED Memory Transfers\n\s*Transfer Size \(Bytes\)\s*Bandwidth\(GB/s\)\n((?:\s*\d+\s+[\d\.]+\n)+)",
            "d2d": r"Device to Device Bandwidth,.*?\n\s*PINNED Memory Transfers\n\s*Transfer Size \(Bytes\)\s*Bandwidth\(GB/s\)\n((?:\s*\d+\s+[\d\.]+\n)+)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.DOTALL)
            if match:
                data_block = match.group(1)
                bw_values = [float(val) for val in re.findall(r"\d+\s+([\d\.]+)", data_block)]
                if bw_values:
                    avg_bw_total = sum(bw_values) / len(bw_values)
                    avg_bw_per_card = avg_bw_total / num_devices
                    results[key] = avg_bw_per_card
    except Exception as e:
        print(f"[Parser Error] Error parsing BandwidthTest output: {e}")
    return results

def parse_p2p_test_output(output: str) -> dict:
    """Parses the output of the p2pBandwidthLatencyTest executable."""
    results = {}
    def calculate_off_diagonal_avg(matrix_block: str) -> float:
        off_diagonal_values = []
        lines = matrix_block.strip().split('\n')[1:]
        for row_index, line in enumerate(lines):
            try:
                values = [float(v) for v in line.split()[1:]]
                for col_index, value in enumerate(values):
                    if row_index != col_index:
                        off_diagonal_values.append(value)
            except (ValueError, IndexError):
                continue
        if off_diagonal_values:
            return sum(off_diagonal_values) / len(off_diagonal_values)
        return 0.0
    try:
        uni_header = "Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)"
        uni_pattern = re.compile(f"^{re.escape(uni_header)}\n((?:^\\s*(?:D.D|\\d).*\\n?)+)", re.MULTILINE)
        uni_match = uni_pattern.search(output)
        if uni_match:
            matrix_data = uni_match.group(1)
            results['unidirectional_avg'] = calculate_off_diagonal_avg(matrix_data)
        bi_header = "Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)"
        bi_pattern = re.compile(f"^{re.escape(bi_header)}\n((?:^\\s*(?:D.D|\\d).*\\n?)+)", re.MULTILINE)
        bi_match = bi_pattern.search(output)
        if bi_match:
            matrix_data = bi_match.group(1)
            results['bidirectional_avg'] = calculate_off_diagonal_avg(matrix_data)
    except Exception as e:
        print(f"[Parser Error] Error parsing P2P-TEST average value: {e}")
    return results

def parse_nccl_test_output(output: str, command: str) -> dict:
    """Parses the output of the NCCL performance tests."""
    results = {}
    try:
        executable_path = command.split()[0]
        executable_name = os.path.basename(executable_path)
        test_type = executable_name.replace('_perf', '')
        results['test_type'] = test_type
        match = re.search(r"# Avg bus bandwidth\s*:\s*([\d\.]+)", output)
        if match:
            results['avg_bus_bw'] = float(match.group(1))
    except Exception as e:
        print(f"[Parser Error] Error parsing NCCL-TEST output: {e}")
    return results

# --- Formatting and Output Functions ---

def print_summary_table(summary_data: dict):
    """Prints a formatted summary table to the console."""
    print("\n" + "="*23 + " Benchmark Summary " + "="*23)
    if 'bandwidth' in summary_data and summary_data['bandwidth']:
        print("\n--- CUDA Bandwidth Test ---")
        bw_data = summary_data['bandwidth']
        print(f"  {'Host to Device (H2D) Avg/Card':<35}: {bw_data.get('h2d', 'N/A'):.2f} GB/s")
        print(f"  {'Device to Host (D2H) Avg/Card':<35}: {bw_data.get('d2h', 'N/A'):.2f} GB/s")
        print(f"  {'Device to Device (D2D) Avg/Card':<35}: {bw_data.get('d2d', 'N/A'):.2f} GB/s")

    if 'p2p' in summary_data and summary_data['p2p']:
        print("\n--- P2P Bandwidth Test ---")
        p2p_data = summary_data['p2p']
        if 'unidirectional_avg' in p2p_data:
            print(f"  {'Unidirectional P2P Bandwidth':<35}: {p2p_data['unidirectional_avg']:.2f} GB/s")
        if 'bidirectional_avg' in p2p_data:
            print(f"  {'Bidirectional P2P Bandwidth':<35}: {p2p_data['bidirectional_avg']:.2f} GB/s")

    nccl_keys = sorted([k for k in summary_data if k.startswith('nccl_')])
    if nccl_keys:
        print("\n--- NCCL Tests ---")
        for key in nccl_keys:
            nccl_data = summary_data[key]
            test_type = nccl_data.get('test_type', 'N/A')
            avg_bw = nccl_data.get('avg_bus_bw', 'N/A')
            if test_type != 'N/A' and isinstance(avg_bw, float):
                print(f"  {test_type.replace('_', ' ').title()}: {avg_bw:.2f} GB/s")
            else:
                print(f"  {test_type.replace('_', ' ').title()}: {avg_bw}")

    print("\n" + "="*65 + "\n")

def write_summary_to_excel(summary_data: dict, filename: str):
    """Writes the summary data to an Excel file."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Benchmark Summary"
    header = ["Test Category", "Test Item", "Result"]
    ws.append(header)

    rows_to_write = []
    if 'bandwidth' in summary_data and summary_data['bandwidth']:
        bw_data = summary_data['bandwidth']
        rows_to_write.extend([
            ("CUDA Bandwidth Test", "Host to Device (H2D) Avg/Card", f"{bw_data.get('h2d', 'N/A'):.2f} GB/s"),
            ("CUDA Bandwidth Test", "Device to Host (D2H) Avg/Card", f"{bw_data.get('d2h', 'N/A'):.2f} GB/s"),
            ("CUDA Bandwidth Test", "Device to Device (D2D) Avg/Card", f"{bw_data.get('d2d', 'N/A'):.2f} GB/s"),
        ])

    if 'p2p' in summary_data and summary_data['p2p']:
        p2p_data = summary_data['p2p']
        rows_to_write.extend([
            ("P2P Bandwidth Test", "Unidirectional P2P Bandwidth", f"{p2p_data.get('unidirectional_avg', 'N/A'):.2f} GB/s"),
            ("P2P Bandwidth Test", "Bidirectional P2P Bandwidth", f"{p2p_data.get('bidirectional_avg', 'N/A'):.2f} GB/s"),
        ])

    nccl_keys = sorted([k for k in summary_data if k.startswith('nccl_')])
    for key in nccl_keys:
        nccl_data = summary_data[key]
        test_type = nccl_data.get('test_type', 'N/A').replace('_', ' ').title()
        avg_bw = nccl_data.get('avg_bus_bw', 'N/A')
        result = f"{avg_bw:.2f} GB/s" if isinstance(avg_bw, float) else avg_bw
        rows_to_write.append(("NCCL Test", test_type, result))

    for row in rows_to_write:
        ws.append(row)

    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        ws.column_dimensions[column].width = adjusted_width

    wb.save(filename)

# --- Main Function ---
def run_nccl_benchmark(args):
    """Runs the full suite of NCCL benchmarks."""
    log_dir_base = getattr(args, 'log_dir', './benchmark_logs')
    log_dir_nccl = os.path.join(log_dir_base, 'nccl')
    os.makedirs(log_dir_nccl, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"nccl_test_{current_time}.log"
    log_filepath = os.path.join(log_dir_nccl, log_filename)
    excel_filename = f"nccl_summary_{current_time}.xlsx"
    excel_filepath = os.path.join(log_dir_nccl, excel_filename)

    # --- Get number of GPUs ---
    num_gpus = 8 # Default value
    try:
        result = subprocess.run("nvidia-smi -L | wc -l", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        gpu_count = int(result.stdout.strip())
        if gpu_count > 0:
            num_gpus = gpu_count
            print(f"Successfully detected {num_gpus} GPUs.")
        else:
            print(f"Could not determine GPU count, defaulting to {num_gpus}.")
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Error detecting number of GPUs: {e}. Defaulting to {num_gpus}.")

    commands_to_execute = [
        {"command": f"{args.BandWidth_Test} --device=all --mode=range --start=64000000 --end=256000000 --increment=64000000", "sheet_name": "BandwidthTest", "parser": parse_bandwidth_test_output, "summary_key": "bandwidth"},
        {"command": args.p2p_test, "sheet_name": "P2P-Bandwidth-Latency-Test", "parser": parse_p2p_test_output, "summary_key": "p2p"},
        {"command": f"{args.All_Reduce_perf} -b 128m -e 8g -f 2 -g {num_gpus}", "sheet_name": "All-Reduce-Perf-Test", "parser": parse_nccl_test_output, "summary_key": "nccl_all_reduce"},
        {"command": f"{args.All_Gather_perf} -b 128m -e 8g -f 2 -g {num_gpus}", "sheet_name": "All-Gather-Perf-Test", "parser": parse_nccl_test_output, "summary_key": "nccl_all_gather"},
    ]

    summary_data = {}

    with open(log_filepath, 'a', encoding='utf-8') as log_f:
        for item in commands_to_execute:
            command = item["command"]
            sheet_name = item["sheet_name"]

            print(f"Executing: {sheet_name}...")
            log_f.write(f"\n{'='*20} [ {sheet_name} ] - Execution Result {'='*20}\n")
            log_f.write(f"Command: {command}\n")
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            print(f"\n{'='*20} [ {sheet_name} ] - Execution Result {'='*20}")
            print(f"Command: {command}")
            print("--- Output Start ---")
            print(result.stdout)
            print(f"--- Output End ---\n{'='*62}\n")
            log_f.write("--- Output Start ---\n")
            log_f.write(result.stdout)
            log_f.write(f"--- Output End ---\n{'='*62}\n\n")

            if item["parser"]:
                parser_func = item["parser"]
                summary_key = item["summary_key"]
                if summary_key.startswith('nccl_'):
                    summary_data[summary_key] = parser_func(result.stdout, command)
                else:
                    summary_data[summary_key] = parser_func(result.stdout)

    print_summary_table(summary_data)
    write_summary_to_excel(summary_data, excel_filepath)
    print(f"Summary information has been saved to: {excel_filepath}")
    return summary_data


# --- Argument Parsing ---
def add_nccl_args(parser):
    """Adds arguments for the NCCL benchmark tools to the argument parser."""
    group = parser.add_argument_group('NCCL Benchmark Paths')
    group.add_argument('-b', '--BandWidth-Test', type=str,
        default='./testtools/bandwidthTest',
        help='Path to the bandwidthTest executable')
    group.add_argument('-p', '--p2p-test', type=str,
        default='./testtools/p2pBandwidthLatencyTest',
        help='Path to the p2pBandwidthLatencyTest executable')
    group.add_argument('-R', '--All_Reduce_perf', type=str,
        default='./testtools/all_reduce_perf',
        help='Path to the all_reduce_perf executable')
    group.add_argument('-G', '--All_Gather_perf', type=str,
        default='./testtools/all_gather_perf',
        help='Path to the all_gather_perf executable')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run NCCL Benchmark Tests')
    add_nccl_args(parser)
    args = parser.parse_args()
    run_nccl_benchmark(args)
