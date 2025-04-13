#!/bin/bash

# 使用说明：
# 本脚本用于在 Ubuntu 系统上自动配置运行环境，包括：
# 1. 配置清华 APT 源（如果未配置）。
# 2. 安装 NVIDIA GPU 驱动、CUDA Toolkit 和 NCCL。
# 3. 安装必要的开发工具（如 GCC、Git、CMake 等）。
# 4. 克隆并编译 NCCL-tests 和 CUDA-samples。

# 使用方法：
# 1. 确保以 root 用户或使用 sudo 权限运行本脚本。
# 2. 运行脚本：
#    chmod +x setup_env.sh
#    sudo ./setup_env.sh
# 3. 脚本会自动检测并安装所需的软件包和工具。
# 4. 如果某些步骤已完成（如清华源已配置或软件已安装），脚本会跳过这些步骤。

# 注意事项：
# 1. 请确保系统已连接网络，且可以访问清华源和 GitHub。
# 2. 脚本默认安装 NVIDIA 驱动版本 570 和 CUDA Toolkit。
# 3. 如果需要调整安装的版本，请修改对应的安装命令。
# 4. 脚本运行完成后，NCCL-tests 和 CUDA-samples 的编译结果会保存在当前目录下。

# 设置脚本在遇到错误时停止执行
set -e

# 检测并配置清华 APT 源
echo "检测并配置清华 APT 源..."
UBUNTU_VERSION=$(lsb_release -cs)  # 获取 Ubuntu 版本代号（如 focal、jammy）
if ! grep -q "mirrors.tuna.tsinghua.edu.cn" /etc/apt/sources.list; then
    echo "配置清华 APT 源..."
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak  # 备份原始 sources.list
    sudo bash -c "cat > /etc/apt/sources.list <<EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $UBUNTU_VERSION main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $UBUNTU_VERSION-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $UBUNTU_VERSION-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ $UBUNTU_VERSION-security main restricted universe multiverse
EOF"
else
    echo "清华 APT 源已配置。"
fi

# 更新系统包列表
echo "更新系统包列表..."
sudo apt update

# 检测并安装 NVIDIA GPU 驱动
if ! dpkg -l | grep -q "nvidia-driver"; then
    echo "安装 NVIDIA GPU 驱动..."
    sudo apt install -y linux-headers-$(uname -r)
    sudo apt install -y nvidia-driver-570
else
    echo "NVIDIA GPU 驱动已安装。"
fi

# 检测并安装 CUDA Toolkit
if ! dpkg -l | grep -q "cuda-toolkit"; then
    echo "安装 CUDA Toolkit..."
    sudo apt install -y cuda-toolkit
else
    echo "CUDA Toolkit 已安装。"
fi

# 检测并安装 NCCL
if ! dpkg -l | grep -q "libnccl2"; then
    echo "安装 NCCL..."
    sudo apt install -y libnccl2 libnccl-dev
else
    echo "NCCL 已安装。"
fi

# 检测并安装 GCC
if ! dpkg -l | grep -q "gcc"; then
    echo "安装 GCC..."
    sudo apt install -y gcc
else
    echo "GCC 已安装。"
fi

# 检测并安装 Git
if ! command -v git &> /dev/null; then
    echo "安装 Git..."
    sudo apt install -y git
else
    echo "Git 已安装。"
fi

# 检测并安装 build-essential
if ! dpkg -l | grep -q "build-essential"; then
    echo "安装 build-essential..."
    sudo apt install -y build-essential
else
    echo "build-essential 已安装。"
fi

# 检测并安装 CMake
if ! dpkg -l | grep -q "cmake"; then
    echo "安装 CMake..."
    sudo apt install -y cmake
else
    echo "CMake 已安装。"
fi

# 克隆并编译 NCCL-tests
if [ ! -d "nccl-tests" ]; then
    echo "克隆 NCCL-tests..."
    git clone https://github.com/NVIDIA/nccl-tests.git
else
    echo "NCCL-tests 已存在，跳过克隆。"
fi
cd nccl-tests
echo "编译 NCCL-tests..."
make CUDA_HOME=/usr/local/cuda
cd ..

# 克隆并编译 CUDA-samples
if [ ! -d "cuda-samples" ]; then
    echo "克隆 CUDA-samples..."
    git clone https://github.com/NVIDIA/cuda-samples.git
else
    echo "CUDA-samples 已存在，跳过克隆。"
fi
cd cuda-samples
mkdir -p build && cd build
echo "编译 CUDA-samples..."
cmake ..
make -j$(nproc)
cd ../..

echo "环境准备完成！"
