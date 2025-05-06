import subprocess
import csv
import os
import sys
import argparse
import socket
import datetime


def get_gpu_info():
    """Retrieve GPU model and count using nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        gpu_models = output.strip().split("\n")
        gpu_count = len(gpu_models)
        gpu_model = gpu_models[0] if gpu_models else "Unknown"
        return gpu_model, gpu_count
    except Exception as e:
        print(f"Error retrieving GPU info: {e}")
        return "Unknown", 0


def get_gpu_vbios_info():
    """Retrieve VBIOS version of all GPUs using nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=vbios_version", "--format=csv,noheader"], text=True)
        vbios_versions = output.strip().split("\n")
        return vbios_versions
    except Exception as e:
        print(f"Error retrieving GPU VBIOS info: {e}")
        return []


def get_gpu_power_info():
    """Retrieve GPU power information and Serial Number using nvidia-smi -q -i <gpu_id>."""
    try:
        power_info = []
        gpu_model, gpu_count = get_gpu_info()  # 获取 GPU 数量

        for gpu_id in range(gpu_count):
            # 针对每个 GPU 单独运行 nvidia-smi -q -i <gpu_id>
            output = subprocess.check_output(["nvidia-smi", "-q", "-i", str(gpu_id)], text=True)
            current_gpu = {
                "gpu_id": gpu_id,
                "serial_number": None,
                "default_power_limit": None,
                "current_power_limit": None,
                "max_power_limit": None
            }

            in_power_readings = False  # 标记是否在 "GPU Power Readings" 部分

            for line in output.splitlines():
                line = line.strip()
                if line.startswith("Serial Number"):
                    current_gpu["serial_number"] = line.split(":")[1].strip()
                elif line.startswith("GPU Power Readings"):
                    in_power_readings = True
                elif line == "" or line.startswith("GPU") or line.startswith("Clocks"):
                    in_power_readings = False
                elif in_power_readings:
                    if line.startswith("Default Power Limit"):
                        current_gpu["default_power_limit"] = line.split(":")[1].strip()
                    elif line.startswith("Current Power Limit"):
                        current_gpu["current_power_limit"] = line.split(":")[1].strip()
                    elif line.startswith("Max Power Limit"):
                        current_gpu["max_power_limit"] = line.split(":")[1].strip()

            power_info.append(current_gpu)

        return power_info
    except Exception as e:
        print(f"Error retrieving GPU power info: {e}")
        return []


def run_nccl_test(test_path, b, e, f, g):
    """Run NCCL test and extract Avg bus bandwidth."""
    try:
        cmd = [test_path, "-b", b, "-e", e, "-f", f, "-g", str(g)]
        print(f"Running command: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, text=True)
        print(output)

        # Extract Avg bus bandwidth
        for line in output.splitlines():
            if "Avg bus bandwidth" in line:
                avg_bus_bw = line.split(":")[-1].strip()
                return float(avg_bus_bw)
    except subprocess.CalledProcessError as e:
        print(f"Error running NCCL test: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def parse_bandwidth_test_output(output, gpu_count):
    """Parse bandwidthTest output to extract the last H2D, D2H, and D2D results."""
    h2d_bandwidth = None
    d2h_bandwidth = None
    d2d_bandwidth = None

    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "Host to Device Bandwidth" in line:
            # Locate the H2D section and extract the last value
            while i < len(lines) and "Bandwidth(GB/s)" not in lines[i]:
                i += 1
            if i + 1 < len(lines):
                # Extract the last line in the H2D section
                while i + 1 < len(lines) and lines[i + 1].strip():
                    i += 1
                h2d_bandwidth = float(lines[i].split()[-1]) / gpu_count  # 最后一个值除以 GPU 数量
        elif "Device to Host Bandwidth" in line:
            # Locate the D2H section and extract the last value
            while i < len(lines) and "Bandwidth(GB/s)" not in lines[i]:
                i += 1
            if i + 1 < len(lines):
                # Extract the last line in the D2H section
                while i + 1 < len(lines) and lines[i + 1].strip():
                    i += 1
                d2h_bandwidth = float(lines[i].split()[-1]) / gpu_count  # 最后一个值除以 GPU 数量
        elif "Device to Device Bandwidth" in line:
            # Locate the D2D section and extract the last value
            while i < len(lines) and "Bandwidth(GB/s)" not in lines[i]:
                i += 1
            if i + 1 < len(lines):
                # Extract the last line in the D2D section
                while i + 1 < len(lines) and lines[i + 1].strip():
                    i += 1
                d2d_bandwidth = float(lines[i].split()[-1]) / gpu_count  # 最后一个值除以 GPU 数量

    return h2d_bandwidth, d2h_bandwidth, d2d_bandwidth


def parse_p2p_bandwidth_latency_test_output(output):
    """Parse p2pBandwidthLatencyTest output to extract P2P Bandwidth results."""
    unidirectional_bandwidth = []
    bidirectional_bandwidth = []

    lines = output.splitlines()
    unidirectional_section = False
    bidirectional_section = False

    for line in lines:
        if "Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)" in line:
            unidirectional_section = True
            bidirectional_section = False
            continue
        elif "Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)" in line:
            unidirectional_section = False
            bidirectional_section = True
            continue

        if unidirectional_section and line.strip():
            unidirectional_bandwidth.append(line.strip())
        elif bidirectional_section and line.strip():
            bidirectional_bandwidth.append(line.strip())

    return unidirectional_bandwidth, bidirectional_bandwidth


def extract_unidirectional_bandwidth(log_file):
    """Extract data between 'Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)'
    and 'Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)', excluding the title line."""
    extracted_data = []
    start_extracting = False

    # 打开日志文件并逐行读取
    with open(log_file, "r") as file:
        for line in file:
            if "Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)" in line:
                start_extracting = True  # 从下一行开始提取
                continue  # 跳过标题行
            elif "Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)" in line:
                break  # 遇到这一行停止提取

            if start_extracting:
                extracted_data.append(line.strip())

    # 验证并返回提取的数据
    if extracted_data:
        return "\n".join(extracted_data)  # 用换行符连接提取的行
    else:
        return "ERROR: Unable to extract Unidirectional P2P=Enabled Bandwidth data."


def extract_bidirectional_bandwidth(log_file):
    """Extract data between 'Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)'
    and 'P2P=Disabled Latency Matrix (us)', excluding the title line."""
    extracted_data = []
    start_extracting = False

    # 打开日志文件并逐行读取
    with open(log_file, "r") as file:
        for line in file:
            if "Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)" in line:
                start_extracting = True  # 从下一行开始提取
                continue  # 跳过标题行
            elif "P2P=Disabled Latency Matrix (us)" in line:
                break  # 遇到这一行停止提取

            if start_extracting:
                extracted_data.append(line.strip())

    # 验证并返回提取的数据
    if extracted_data:
        return "\n".join(extracted_data)  # 用换行符连接提取的行
    else:
        return "ERROR: Unable to extract Bidirectional P2P=Enabled Bandwidth data."


def write_to_csv(hostname, gpu_model, results, vbios_versions, output_file="nccl_results.csv"):
    """Write results and VBIOS info to a CSV file."""
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write NCCL test results
        writer.writerow([f"{hostname}-{results['gpu_count']} GPUs-{gpu_model}"])
        writer.writerow(["Test Type", "Command", "2 GPUs", "4 GPUs", "8 GPUs"])
        for test_type, data in results["tests"].items():
            # Combine all tested GPU counts into a single Command string
            tested_gpus = "/".join([str(g) for g in [2, 4, 8] if g <= results["gpu_count"]])
            # Remove any existing -g argument from the original command
            command_parts = data["command"].split()
            if "-g" in command_parts:
                g_index = command_parts.index("-g")
                command_parts = command_parts[:g_index]  # Remove -g and its value
            command_with_gpus = " ".join(command_parts) + f" -g {tested_gpus}"
            row = [test_type, command_with_gpus]
            for gpus in [2, 4, 8]:
                if gpus <= results["gpu_count"]:
                    row.append(data.get(f"{gpus} GPUs", "N/A"))
                else:
                    row.append("N/A")
            writer.writerow(row)

        # Write a blank line to separate tables
        writer.writerow([])

        # Write GPU VBIOS info
        writer.writerow(["GPU ID", "VBIOS Version"])
        for idx, vbios in enumerate(vbios_versions):
            writer.writerow([f"GPU {idx}", vbios])

    print(f"Results written to {output_file}")


def write_bandwidth_test_results(command, h2d, d2h, d2d, output_file="bandwidth_test_results.csv"):
    """Write bandwidthTest results to a CSV file."""
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Test-Type", "Command", "H2D", "D2H", "D2D"])
        writer.writerow(["bandwidthTest", command, h2d, d2h, d2d])
    print(f"Bandwidth test results written to {output_file}")


def write_combined_results(hostname, gpu_model, nccl_results, vbios_versions, bandwidth_results, p2p_results, power_info=None):
    """Write combined NCCL, bandwidthTest, P2P Bandwidth Latency Test, and GPU info results to a CSV file."""
    # 获取当前日期和时间并生成文件名
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{gpu_model.replace(' ', '_')}-{datetime_str}.csv"

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # 写入主机名和 GPU 信息
        writer.writerow([f"{hostname}-{nccl_results['gpu_count']} GPUs-{gpu_model}"])

        # 写入空行分隔表格
        writer.writerow([])

        # 写入 VBIOS 和 GPU 功率信息到同一个表
        writer.writerow(["GPU ID", "VBIOS Version", "Serial Number (SN)", "Default Power Limit", "Current Power Limit", "Max Power Limit"])
        for idx in range(len(vbios_versions)):
            vbios = vbios_versions[idx] if idx < len(vbios_versions) else "N/A"
            power = power_info[idx] if power_info and idx < len(power_info) else {}
            writer.writerow([
                idx,
                vbios,
                power.get("serial_number", "N/A"),
                power.get("default_power_limit", "N/A"),
                power.get("current_power_limit", "N/A"),
                power.get("max_power_limit", "N/A")
            ])

        # 写入空行分隔表格
        writer.writerow([])

        # 写入 NCCL 测试结果
        writer.writerow(["Test Type", "Command", "2 GPUs", "4 GPUs", "8 GPUs"])
        for test_type, data in nccl_results["tests"].items():
            row = [test_type, data["command"]]
            for gpus in [2, 4, 8]:
                row.append(data.get(f"{gpus} GPUs", "N/A"))
            writer.writerow(row)

        # 写入空行分隔表格
        writer.writerow([])

        # 写入 bandwidthTest 测试结果
        if bandwidth_results:
            writer.writerow(["Test-Type", "Command", "H2D", "D2H", "D2D"])
            writer.writerow([
                "bandwidthTest",
                bandwidth_results["command"],
                bandwidth_results["h2d"],
                bandwidth_results["d2h"],
                bandwidth_results["d2d"]
            ])

        # 写入空行分隔表格
        writer.writerow([])

        # 写入 p2pBandwidthLatencyTest 测试结果
        if p2p_results:
            # 从日志文件中提取 Unidirectional 数据
            unidirectional_data = extract_unidirectional_bandwidth(p2p_results["log_file"])

            # 从日志文件中提取 Bidirectional 数据
            bidirectional_data = extract_bidirectional_bandwidth(p2p_results["log_file"])

            # 写入结果
            writer.writerow(["test-type", "Command", "Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)", "Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)"])
            writer.writerow([
                "p2pBandwidthLatencyTest",
                p2p_results["command"],
                unidirectional_data,  # 写入 Unidirectional 数据
                bidirectional_data    # 写入 Bidirectional 数据
            ])

        # 写入空行分隔表格
        writer.writerow([])

        # 写入 nvidia-smi topo -m 输出
        try:
            topo_output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
            writer.writerow(["NVIDIA-SMI Topology"])
            writer.writerow([topo_output])  # 将整个输出作为单个单元格写入
        except Exception as e:
            writer.writerow(["NVIDIA-SMI Topology", f"Error retrieving topology: {e}"])

    print(f"Combined results written to {output_file}")


def run_bandwidth_test(test_path, device="all", mode="range", start="64000000", end="256000000", increment="32000000"):
    """Run bandwidthTest and extract results."""
    try:
        cmd = [
            test_path,
            f"--device={device}",
            f"--mode={mode}",
            f"--start={start}",
            f"--end={end}",
            f"--increment={increment}"
        ]
        print(f"Running command: {' '.join(cmd)}")
        output = subprocess.check_output(cmd, text=True)
        print(output)
        return output  # Return the raw output for further processing
    except subprocess.CalledProcessError as e:
        print(f"Error running bandwidthTest: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def run_p2p_bandwidth_latency_test(test_path, device="all", mode="range", start="64000000", end="256000000", increment="32000000"):
    """Run p2pBandwidthLatencyTest and save results to a log file."""
    try:
        # 构建命令
        cmd = [
            test_path,
            f"--device={device}",
            f"--mode={mode}",
            f"--start={start}",
            f"--end={end}",
            f"--increment={increment}"
        ]
        print(f"Running command: {' '.join(cmd)}")
        
        # 执行命令并获取输出
        output = subprocess.check_output(cmd, text=True)

        # 获取当前日期和时间并生成日志文件名
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"p2pBandwidthLatencyTest-{datetime_str}.log"

        # 将输出保存到日志文件
        with open(log_filename, "w") as log_file:
            log_file.write(output)
        print(f"p2pBandwidthLatencyTest output saved to {log_filename}")

        # 返回命令和日志文件路径
        return {
            "command": " ".join(cmd),
            "log_file": log_filename,  # 返回日志文件路径
            "raw_output": output  # 返回完整的原始输出
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running p2pBandwidthLatencyTest: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="NCCL and CUDA Sample Test Script")
    parser.add_argument("--nccl-test-path", type=str, 
                        default="./nccl-tests/build/",  # 设置默认值
                        help="Path to the NCCL test binaries directory (default: %(default)s)")
    parser.add_argument("--bandwidth-test-path", type=str, 
                        default="./cuda-samples/build/Samples/1_Utilities/bandwidthTest/bandwidthTest",
                        help="Path to the bandwidthTest binary (default: %(default)s)")
    parser.add_argument("--p2pBandwidthLatencyTest-path", type=str, 
                        default="./cuda-samples/build/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest",  # 设置默认值
                        help="Path to the p2pBandwidthLatencyTest binary (default: %(default)s)")
    parser.add_argument("-b", type=str, default="6G", help="Minimum size (default: 6G, for nccl-test)")
    parser.add_argument("-e", type=str, default="24G", help="Maximum size (default: 24G, for nccl-test)")
    parser.add_argument("-f", type=str, default="2", help="Step factor (default: 2, for nccl-test)")
    parser.add_argument("-g", type=str, default="8", help="Number of GPUs to test (default: 8, for nccl-test)")
    parser.add_argument("--device", type=str, default="all", help="Device to test (default: all, for bandwidthTest and p2pBandwidthLatencyTest)")
    parser.add_argument("--mode", type=str, default="range", help="Mode to test (default: range, for bandwidthTest and p2pBandwidthLatencyTest)")
    parser.add_argument("--start", type=str, default="64000000", help="Start size (default: 64000000, for bandwidthTest and p2pBandwidthLatencyTest)")
    parser.add_argument("--end", type=str, default="256000000", help="End size (default: 256000000, for bandwidthTest and p2pBandwidthLatencyTest)")
    parser.add_argument("--increment", type=str, default="32000000", help="Increment size (default: 32000000, for bandwidthTest and p2pBandwidthLatencyTest)")
    parser.add_argument("--vbios", action="store_true", help="Enable VBIOS detection (default: disabled)")
    parser.add_argument("--powerinfo", action="store_true", help="Enable GPU power info extraction (default: disabled)")
    args = parser.parse_args()

    hostname = socket.gethostname()
    gpu_model, gpu_count = get_gpu_info()

    # Initialize results
    vbios_versions = []
    results = {
        "gpu_count": gpu_count,
        "tests": {}
    }
    bandwidth_results = None
    p2p_results = None
    power_info = None

    # Check if VBIOS detection is enabled
    if args.vbios:
        vbios_versions = get_gpu_vbios_info()

    # Check if GPU power info extraction is enabled
    if args.powerinfo:
        power_info = get_gpu_power_info()

    # Run NCCL tests if the path is provided
    if args.nccl_test_path:
        max_gpus_to_test = int(args.g)
        gpu_test_range = [g for g in [2, 4, 8] if g <= max_gpus_to_test]

        nccl_tests = ["all_reduce_perf", "all_gather_perf"]

        for test_binary in nccl_tests:
            test_path = os.path.join(args.nccl_test_path, test_binary)
            for gpus in gpu_test_range:
                if gpus > gpu_count:
                    continue
                avg_bus_bw = run_nccl_test(test_path, args.b, args.e, args.f, gpus)
                if avg_bus_bw is not None:
                    if test_binary not in results["tests"]:
                        results["tests"][test_binary] = {
                            "command": f"{test_path} -b {args.b} -e {args.e} -f {args.f} -g {gpus}",
                            "2 GPUs": None,
                            "4 GPUs": None,
                            "8 GPUs": None
                        }
                    results["tests"][test_binary][f"{gpus} GPUs"] = avg_bus_bw

    # Run Bandwidth test if the path is provided
    if args.bandwidth_test_path:
        output = run_bandwidth_test(
            args.bandwidth_test_path,
            device=args.device,
            mode=args.mode,
            start=args.start,
            end=args.end,
            increment=args.increment
        )
        if output:
            h2d, d2h, d2d = parse_bandwidth_test_output(output, gpu_count)  # 传递 GPU 数量
            bandwidth_results = {
                "command": f"{args.bandwidth_test_path} --device={args.device} --mode={args.mode} --start={args.start} --end={args.end} --increment={args.increment}",
                "h2d": h2d,
                "d2h": d2h,
                "d2d": d2d
            }

    # Run P2P Bandwidth Latency test if the path is provided
    if args.p2pBandwidthLatencyTest_path:
        p2p_results = run_p2p_bandwidth_latency_test(
            args.p2pBandwidthLatencyTest_path,
            device=args.device,
            mode=args.mode,
            start=args.start,
            end=args.end,
            increment=args.increment
        )

    # Write combined results to CSV
    write_combined_results(hostname, gpu_model, results, vbios_versions, bandwidth_results, p2p_results, power_info)


if __name__ == "__main__":
    main()