# Docker 使用指南

## 前提条件

⚠️ **重要**：本项目需要 NVIDIA GPU 支持。Docker 可以在以下环境中运行：
- **Linux 系统**（推荐，最简单）
- **Windows + WSL2**（需要额外配置，见下方说明）

---

## Windows 用户指南

### 在 Windows 上使用 Docker

✅ **可以**在 Windows 上运行，但需要以下配置：

#### 步骤1：安装 WSL2

1. **启用 WSL 功能**（以管理员身份运行 PowerShell）：
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

2. **安装 WSL2 内核更新包**：
   - 从 [Microsoft 官网](https://aka.ms/wsl2kernel) 下载并安装

3. **设置默认 WSL 版本为 2**：
   ```powershell
   wsl --set-default-version 2
   ```

4. **安装 Ubuntu（或其他Linux发行版）**：
   ```powershell
   wsl --install -d Ubuntu
   ```

#### 步骤2：安装 Docker Desktop

1. 从 [Docker 官网](https://www.docker.com/products/docker-desktop/) 下载并安装 Docker Desktop

2. **配置 Docker Desktop**：
   - 打开 Docker Desktop
   - 进入 **Settings** > **General**
   - 确保选中 **"Use the WSL 2 based engine"**
   - 进入 **Settings** > **Resources** > **WSL Integration**
   - 启用与你的 WSL2 发行版（Ubuntu）的集成

#### 步骤3：安装 NVIDIA GPU 驱动（支持 WSL2）

1. **安装支持 WSL2 的 NVIDIA 驱动**：
   - 从 [NVIDIA 官网](https://developer.nvidia.com/cuda/wsl) 下载并安装
   - ⚠️ 这是**Windows驱动**，不是Linux驱动
   - 安装后重启电脑

2. **在 WSL2 中验证 GPU**：
   ```bash
   # 在 WSL2 Ubuntu 终端中
   nvidia-smi
   ```
   如果能看到GPU信息，说明配置成功。

#### 步骤4：在 WSL2 中安装 NVIDIA Container Toolkit

在 WSL2 Ubuntu 终端中执行：

```bash
# 安装 Docker（如果Docker Desktop未自动配置）
# 通常Docker Desktop会自动配置，跳过此步

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker  # 如果systemd可用
```

#### 步骤5：验证配置

在 WSL2 Ubuntu 终端中：

```bash
# 检查GPU
nvidia-smi

# 测试Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

如果两个命令都能显示GPU信息，说明配置成功！

#### 步骤6：在 WSL2 中运行项目

```bash
# 在 WSL2 Ubuntu 终端中，进入项目目录
cd /mnt/c/Users/Administrator/Documents/GitHub/PolyU_COMP_Final_Year_Project_2026_Spring

# 按照下面的Linux步骤操作
cd docker
chmod +x run.sh
./run.sh build
./run.sh run
```

⚠️ **注意事项**：
- 所有Docker命令需要在 **WSL2 终端**中运行，不是在Windows PowerShell
- 项目路径可以使用 `/mnt/c/...` 访问Windows文件系统
- 或者将项目克隆到WSL2文件系统中（`~/projects/...`）以获得更好性能

---

## Linux 用户指南

### 1. 安装 Docker（如果未安装）

```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### 2. 安装 NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. 验证安装

```bash
# 检查Docker
docker --version

# 检查NVIDIA支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 快速开始（Linux 和 Windows WSL2 通用）

## 快速开始

### 步骤1：构建Docker镜像

```bash
cd docker
chmod +x run.sh
./run.sh build
```

这会自动：
- 从GitHub克隆cuSZp仓库
- 编译cuSZp库
- 编译cuSZp包装器
- 安装所有Python依赖

**预计时间**：10-30分钟（取决于网络和CPU速度）

### 步骤2：运行容器

```bash
./run.sh run
```

这会启动一个交互式容器，你可以直接在容器内工作。

### 步骤3：在容器内运行项目

进入容器后，你已经在 `/workspace` 目录中，所有项目文件都在这里：

```bash
# 验证编译结果
ls -la integration/cuszp_wrapper/build/
ls -la cuSZp/install/lib/

# 运行基线性能测试
python3 benchmarks/baseline_profiling.py

# 运行压缩性能测试
python3 benchmarks/compression_benchmark.py

# 设置Python路径（如果需要）
export PYTHONPATH=$PYTHONPATH:/workspace/integration/compression_pipeline
```

## 其他运行方式

### 方法2：使用 docker-compose

```bash
cd docker

# 构建镜像
docker-compose build

# 后台运行容器
docker-compose up -d

# 进入容器
docker-compose exec vllm-cuszp bash

# 停止容器
docker-compose down
```

### 方法3：直接使用 docker 命令

```bash
# 构建镜像
cd docker
docker build -t vllm-cuszp:latest -f Dockerfile ..

# 运行容器
docker run --gpus all -it --rm \
    -v $(pwd)/..:/workspace \
    -w /workspace \
    vllm-cuszp:latest
```

### 方法4：运行测试（不进入容器）

```bash
cd docker
./run.sh test
```

这会自动运行所有测试脚本。

## run.sh 脚本命令说明

| 命令 | 说明 |
|------|------|
| `./run.sh build` | 构建Docker镜像 |
| `./run.sh run` | 运行交互式容器 |
| `./run.sh exec` | 进入运行中的容器 |
| `./run.sh test` | 运行测试脚本 |
| `./run.sh compose-build` | 使用docker-compose构建 |
| `./run.sh compose-up` | 使用docker-compose启动（后台） |
| `./run.sh compose-down` | 停止docker-compose容器 |

## 常见问题

### Q: 构建镜像时出错怎么办？

A: 
1. 检查网络连接（需要从GitHub克隆cuSZp）
2. 确保有足够的磁盘空间（至少10GB）
3. 检查Docker是否有足够权限：`sudo usermod -aG docker $USER`（需要重新登录）

### Q: 容器内无法使用GPU？

A: 
1. 确保安装了NVIDIA Container Toolkit
2. 检查GPU是否可见：`nvidia-smi`
3. 在容器内运行：`nvidia-smi` 验证

### Q: 如何修改代码后重新测试？

A: 
- 项目目录已挂载到容器，修改代码后直接在容器内重新运行即可
- 如果修改了C++代码，需要重新编译：
  ```bash
  cd /workspace/integration/cuszp_wrapper/build
  make -j$(nproc)
  ```

### Q: 如何保存容器中的更改？

A: 
- 代码更改会自动保存（因为目录已挂载）
- 如果需要保存容器状态，使用 `docker commit` 或重新构建镜像

## 下一步

- 查看 `../docs/DEPLOYMENT.md` 了解详细部署选项
- 查看 `../docs/INTEGRATION_GUIDE.md` 了解如何集成到vLLM
- 查看 `../README.md` 了解项目概述

