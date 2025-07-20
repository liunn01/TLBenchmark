import os
import pandas as pd
import subprocess
from datetime import datetime
import re

def add_baseinfo_args(parser):
    pass

def run_and_log(cmd, log_path, desc=None, shell=False):
    cmd_str = desc or (cmd if isinstance(cmd, str) else ' '.join(cmd))
    print(f'==== {cmd_str} ===')
    try:
        result = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            output = f"Error: {result.stderr.strip()}"
        else:
            output = result.stdout.strip()
    except Exception as e:
        output = f"Exception: {e}"
    print(output)
    with open(log_path, 'a') as f:
        f.write(f"==== {cmd_str} ===\n\n{output}\n\n")
    return output

def run_baseinfo(args):
    """
    执行baseinfo测试，将所有命令的原始文本结果都写入log文件并打印到console，详细GPU表返回DataFrame。
    """
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f'baseinfo-{timestamp}.log')

    # 1. lscpu | grep Arch -A24
    run_and_log("lscpu | grep Arch -A24", log_path, desc="lscpu | grep Arch -A24", shell=True)
    # 2. dmidecode -t memory | grep -E 'Manufacturer:|Part Number|Size|Speed|Configured Memory Speed' | grep -Ev 'None|Not|Unknown|No Module Installed'
    run_and_log("dmidecode -t memory | grep -E 'Manufacturer:|Part Number|Size|Speed|Configured Memory Speed' | grep -Ev 'None|Not|Unknown|No Module Installed'", log_path, desc="dmidecode -t memory ...", shell=True)
    # 3. free -h
    run_and_log(["free", "-h"], log_path, desc="free -h")
    # 4. head -n 4 /etc/os-release
    run_and_log(["head", "-n", "4", "/etc/os-release"], log_path, desc="head -n 4 /etc/os-release")
    # 5. uname -sr
    run_and_log(["uname", "-sr"], log_path, desc="uname -sr")
    # 6. nvcc -V
    run_and_log(["nvcc", "-V"], log_path, desc="nvcc -V")
    # 7. nvidia-smi --query-gpu=driver_version --format=csv,noheader
    run_and_log(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], log_path, desc="nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    # nvidia-smi topo -m
    topo_output = run_and_log(["nvidia-smi", "topo", "-m"], log_path, desc="nvidia-smi topo -m")
    # 只在表格部分查找 NVLink
    has_nvlink = False
    if isinstance(topo_output, str):
        table_part = topo_output.split("Legend")[0]
        if re.search(r'\bNV[12468]\b', table_part):
            has_nvlink = True

    if has_nvlink:
        # NVLink: 精简字段
        smi_cmd = [
            'nvidia-smi',
            '--query-gpu=index,vbios_version,gpu_uuid,clocks.max.sm,clocks.max.mem,enforced.power.limit,memory.total',
            '--format=csv,noheader,nounits'
        ]
        smi_desc = 'nvidia-smi --query-gpu=index,vbios_version,gpu_uuid,clocks.max.sm,clocks.max.mem,enforced.power.limit,memory.total --format=csv,noheader,nounits'
        header = [
            'index', 'vbios_version', 'gpu_uuid',
            'clocks.max.sm', 'clocks.max.mem', 'enforced.power.limit', 'memory.total'
        ]
    else:
        # PCIe: 原有字段
        smi_cmd = [
            'nvidia-smi',
            '--query-gpu=index,vbios_version,gpu_uuid,pci.bus_id,pci.device_id,pcie.link.width.max,pcie.link.width.current,clocks.max.sm,clocks.max.mem,enforced.power.limit,memory.total',
            '--format=csv,noheader,nounits'
        ]
        smi_desc = 'nvidia-smi --query-gpu=index,vbios_version,gpu_uuid,pci.bus_id,pci.device_id,pcie.link.width.max,pcie.link.width.current,clocks.max.sm,clocks.max.mem,enforced.power.limit,memory.total --format=csv,noheader,nounits'
        header = [
            'index', 'vbios_version', 'gpu_uuid', 'pci.bus_id', 'pci.device_id',
            'pcie.link.width.max', 'pcie.link.width.current',
            'clocks.max.sm', 'clocks.max.mem', 'enforced.power.limit', 'memory.total'
        ]
    smi_text = run_and_log(smi_cmd, log_path, desc=smi_desc)
    if smi_text.startswith("Error") or smi_text.startswith("Exception"):
        df = pd.DataFrame(columns=header)
    else:
        lines = smi_text.split('\n')
        data = [line.split(', ') for line in lines if line.strip()]
        df = pd.DataFrame(data, columns=header)

    # 返回DataFrame而不是写入文件
    return df 