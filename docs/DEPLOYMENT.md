# 部署指南

## 部署方案

本项目需要在支持NVIDIA GPU的Linux环境中运行。以下是几种可行的部署方案：

### 方案1: 使用云GPU服务（推荐）

#### AWS EC2 GPU实例

1. **创建EC2实例**
   - 选择实例类型：`g4dn.xlarge` 或更高（包含NVIDIA T4 GPU）
   - AMI：选择预装CUDA的Deep Learning AMI（如：Deep Learning AMI (Ubuntu 20.04)）

2. **连接到实例**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **验证CUDA安装**
   ```bash
   nvidia-smi
   nvcc --version
   ```

#### Google Cloud Platform (GCP)

1. **创建GPU实例**
   ```bash
   gcloud compute instances create gpu-instance \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --maintenance-policy=TERMINATE \
     --restart-on-failure
   ```

2. **安装CUDA**
   ```bash
   # 按照NVIDIA官方指南安装CUDA
   ```

#### Azure GPU虚拟机

1. 在Azure Portal创建NC系列虚拟机（包含NVIDIA GPU）
2. 选择Ubuntu 20.04 LTS镜像
3. 安装NVIDIA驱动和CUDA

### 方案2: Docker容器（需要Linux环境）

> **快速开始**：如果只需要快速运行，可以使用 `docker/run.sh` 脚本。详细说明请参考 `README.md` 中的"Docker运行"部分。

#### 使用NVIDIA Container Toolkit

1. **在Linux服务器上安装Docker和NVIDIA Container Toolkit**
   ```bash
   # 安装Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # 安装NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **构建Docker镜像**

   **方法A：使用运行脚本（最简单，推荐）**
   ```bash
   cd docker
   chmod +x run.sh
   ./run.sh build
   ```

   **方法B：使用docker build**
   ```bash
   cd docker
   docker build -t vllm-cuszp:latest -f Dockerfile ..
   ```
   注意：`-f Dockerfile` 指定Dockerfile路径，`..` 是构建上下文（项目根目录）

   **方法C：使用docker-compose**
   ```bash
   cd docker
   docker-compose build
   ```

3. **运行容器**

   **方法A：使用运行脚本（最简单，推荐）**
   ```bash
   cd docker
   ./run.sh run
   ```

   **方法B：使用docker run**
   ```bash
   docker run --gpus all -it \
     -v $(pwd)/..:/workspace \
     vllm-cuszp:latest
   ```

   **方法C：使用docker-compose**
   ```bash
   cd docker
   docker-compose up -d  # 后台运行
   docker-compose exec vllm-cuszp bash  # 进入容器
   # 或者直接交互式运行
   docker-compose run --rm vllm-cuszp bash
   ```

4. **在容器内运行测试**
   ```bash
   # 进入容器后
   cd /workspace
   
   # 运行基线性能测试
   python3 benchmarks/baseline_profiling.py
   
   # 运行压缩性能测试
   python3 benchmarks/compression_benchmark.py
   ```

### 方案3: 本地Linux服务器

如果你有访问Linux服务器的权限：

1. **SSH连接到服务器**
2. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd FYP
   ```

3. **按照快速开始指南设置环境**

## 开发工作流建议

### 开发阶段可以完成的工作

1. **代码编写和设计**
   - 集成代码框架
   - 文档编写
   - 测试脚本编写

2. **代码审查**
   - 使用Git进行版本控制
   - 代码审查和重构

### 在GPU环境中完成的工作

1. **编译和测试**
   - cuSZp编译
   - vLLM集成编译
   - 性能测试

2. **调试和优化**
   - 使用Nsight Systems进行性能分析
   - 调试CUDA代码

## 推荐的开发流程

1. **开发阶段**
   - 编写代码框架
   - 编写单元测试（不依赖GPU的部分）
   - 提交到Git

2. **测试阶段**（GPU服务器）
   - 拉取最新代码
   - 编译和运行
   - 收集性能数据

3. **迭代优化**
   - 分析性能数据
   - 优化代码
   - 重复步骤1-2

## 环境配置检查清单

- [ ] Linux操作系统（Ubuntu 20.04+）
- [ ] NVIDIA GPU（支持CUDA 11.0+）
- [ ] NVIDIA驱动已安装（`nvidia-smi`可运行）
- [ ] CUDA Toolkit 11.8+已安装
- [ ] Python 3.9+已安装
- [ ] CMake 3.21+已安装
- [ ] Git已安装
- [ ] 足够的磁盘空间（>100GB用于模型和数据集）

