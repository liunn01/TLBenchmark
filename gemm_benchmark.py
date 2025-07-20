import os
import sys
import subprocess
import csv
from datetime import datetime

def run_gemm_benchmark(args):
    """
    在所有可用的GPU上运行CUTLASS Profiler，并合并结果。
    通过 args.kernels, args.profiler_path, args.output_file, args.log_dir 获取参数。
    输出文件自动存放到 log_dir 子目录下。
    """
    # 处理 log_dir 子目录
    log_dir = getattr(args, 'log_dir', './benchmark_logs_client_only')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name, extension = os.path.splitext(args.output_file)
    final_output_file_with_timestamp = f"{base_name}_{timestamp}{extension}"
    final_output_path = os.path.join(log_dir, final_output_file_with_timestamp)

    if not os.path.isfile(args.profiler_path) or not os.access(args.profiler_path, os.X_OK):
        print(f"错误: Profiler路径 '{args.profiler_path}' 不存在或文件不可执行。")
        print("请使用 --profiler-path=... 参数指定正确的路径。")
        sys.exit(1)

    print(f"将要测试的内核模式: {args.kernels}")
    print(f"Profiler 执行路径: {args.profiler_path}")
    print(f"最终输出文件: {final_output_path}")
    print("-" * 50)

    gpu_indices = get_gpu_indices()
    if not gpu_indices:
        print("错误：未检测到任何NVIDIA GPU。")
        sys.exit(1)
    print(f"将在以下GPU上运行测试: {', '.join(gpu_indices)}")
    print("-" * 50)

    temp_files_map = {}
    temp_file_base_name = os.path.join(log_dir, "temp_profiler_results")

    try:
        for gpu_id in gpu_indices:
            print(f"正在为 GPU {gpu_id} 启动测试...")
            temp_output_base = f"{temp_file_base_name}_gpu_{gpu_id}"
            actual_temp_file = f"{temp_output_base}.gemm.csv"
            temp_files_map[gpu_id] = actual_temp_file
            command = [
                args.profiler_path,
                "--operation=Gemm",
                "--m=8192",
                "--n=8192",
                "--k=8192",
                "--beta=1",
                "--profiling-iterations=10",
                "--providers=cutlass",
                f"--devices={gpu_id}",
                f"--kernels={args.kernels}",
                f"--output={temp_output_base}",
            ]
            print(f"执行命令: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
            if result.stdout:
                print("--- Profiler STDOUT ---")
                print(result.stdout)
            if result.stderr:
                print("--- Profiler STDERR ---")
                print(result.stderr)
            if result.returncode != 0:
                print(f"警告：GPU {gpu_id} 上的 cutlass_profiler 运行失败，返回码: {result.returncode}。")
            else:
                print(f"GPU {gpu_id} 测试运行完成。")
            print("-" * 50)
        consolidate_results(temp_files_map, final_output_path)
    finally:
        print("\n正在清理临时文件...")
        for temp_file in temp_files_map.values():
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    print(f"删除临时文件 {temp_file} 时出错: {e}")
        print("清理完毕。")

def get_gpu_indices():
    """使用nvidia-smi获取所有可用的GPU索引"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        gpu_indices = result.stdout.strip().split('\n')
        if not gpu_indices or not gpu_indices[0]:
            return []
        return gpu_indices
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: `nvidia-smi` 命令执行失败。请确保NVIDIA驱动已正确安装。")
        sys.exit(1)

def consolidate_results(temp_files: dict, final_output_file: str):
    """合并所有临时的CSV文件到一个最终文件中"""
    print(f"\n所有GPU测试运行完毕，开始合并结果到 {final_output_file} ...")
    header_written = False
    import csv
    with open(final_output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        for gpu_id, temp_file_path in temp_files.items():
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                print(f"信息：未找到 GPU {gpu_id} 的有效结果文件 ({temp_file_path})，跳过合并。")
                continue
            with open(temp_file_path, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                try:
                    original_header = next(reader)
                    if not header_written:
                        writer.writerow(['GPU_ID'] + original_header)
                        header_written = True
                        print(f"已创建最终结果文件并写入表头: {final_output_file}")
                    for row in reader:
                        writer.writerow([gpu_id] + row)
                except StopIteration:
                    print(f"警告: GPU {gpu_id} 的结果文件 {temp_file_path} 为空或只有表头。")
    if not header_written:
        print("错误：所有GPU测试均未成功生成结果文件，无法创建最终报告。")
        sys.exit(1)
    print(f"所有结果已成功合并到: {final_output_file}")

def add_gemm_args(parser):
    group = parser.add_argument_group('GEMM Benchmark')
    group.add_argument('-k', '--kernels', type=str, required=True, help='Kernel name or pattern to test. Example: cutlass_tensorop_h16816gemm_*')
    group.add_argument('-p', '--profiler-path', type=str, default='/root/cutlass/build/tools/profiler/cutlass_profiler', help='Path to the cutlass_profiler executable')
    group.add_argument('-O', '--output-file', type=str, default='all_gpus_cutlass_gemm.csv', help='Final merged result CSV file name')
    # 可保留原有参数
    # group.add_argument('--gemm-size', type=int, default=4096, help='(GEMM) Matrix size for GEMM test (example parameter).') 