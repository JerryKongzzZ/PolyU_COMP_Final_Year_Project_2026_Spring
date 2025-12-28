# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 项目概述

本项目旨在将cuSZp压缩库集成到vLLM框架中，通过压缩-传输-解压缩的工作流程来优化CPU-GPU之间的数据传输，从而减少PCIe带宽瓶颈对LLM推理性能的影响。

## 项目结构

```
FYP/
├── cuSZp/                    # cuSZp压缩库源代码
├── vllm_modified/            # 修改后的vLLM代码（待创建）
├── integration/              # 集成代码
│   ├── cuszp_wrapper/       # cuSZp C++包装器
│   └── compression_pipeline/ # 压缩传输管道
├── benchmarks/              # 性能测试脚本
├── docs/                    # 文档
└── docker/                  # Docker部署配置

```

## 环境要求

本项目需要NVIDIA GPU和CUDA支持。

### 支持的环境

1. **Linux系统**（推荐，最简单）
   - Ubuntu 20.04+ 或其他Linux发行版
   - 直接支持Docker和NVIDIA Container Toolkit

2. **Windows系统**（需要WSL2）
   - Windows 10/11 + WSL2 + Docker Desktop
   - 需要安装支持WSL2的NVIDIA驱动
   - 需要在WSL2中配置NVIDIA Container Toolkit
   - 详细步骤请参考 `docker/README.md` 中的"Windows用户指南"

### 部署方案

1. **云GPU服务**：使用AWS、GCP、Azure等云服务商的GPU实例
2. **Docker容器**：使用Docker运行支持CUDA的容器（Linux或Windows WSL2）
3. **本地服务器**：在具有NVIDIA GPU的Linux服务器上运行

详细部署指南请参考 `docs/DEPLOYMENT.md` 和 `docker/README.md`

## 实施阶段

### Phase 1: 环境分析和基线设置 (Weeks 1-3) ✅
- [x] 分析vLLM代码结构
- [x] 了解CPU-GPU数据传输路径
- [x] 设计集成方案
- [x] 创建项目文档

### Phase 2: cuSZp集成准备 (Weeks 4-6) ✅
- [x] 分析cuSZp API
- [x] 创建C++包装器
- [x] 创建Python绑定
- [x] 创建构建系统

### Phase 3: 实现压缩数据传输管道 (Weeks 7-11) ✅
- [x] 实现D2H压缩传输框架
- [x] 实现H2D解压缩传输框架
- [x] 多流重叠和内存管理设计
- [x] 错误处理和回退机制

### Phase 4: 评估和分析 (Weeks 12-15) ✅
- [x] 创建性能基准测试脚本
- [x] 创建压缩性能测试脚本
- [x] 创建Docker部署配置
- [ ] 在GPU环境中运行测试（需要在GPU环境中完成）

## 快速开始

### 1. 环境要求

- Linux系统（Ubuntu 20.04+）
- NVIDIA GPU（支持CUDA 11.0+）
- CUDA Toolkit 11.8+
- Python 3.9+
- CMake 3.21+

### 2. 编译项目

**推荐：使用自动化脚本**（最简单）

```bash
chmod +x build.sh
./build.sh
```

该脚本会自动编译cuSZp和cuSZp包装器。

**手动编译**（如需自定义选项）：

详细步骤请参考 `docs/INTEGRATION_GUIDE.md` 中的"构建步骤"部分。

### 3. 使用Docker运行（推荐）

**快速开始**：
```bash
cd docker
chmod +x run.sh
./run.sh build    # 构建镜像
./run.sh run      # 运行容器
```

详细说明和更多选项请参考 `docs/DEPLOYMENT.md` 中的"Docker容器"部分。

### 4. 集成到vLLM

详细步骤请参考 `docs/INTEGRATION_GUIDE.md`

简要步骤：
1. 修改vLLM的`platforms/cuda.py`文件
2. 导入压缩交换管理器
3. 在初始化时启用压缩功能

### 5. 运行测试

```bash
# 基线性能测试
python benchmarks/baseline_profiling.py

# 压缩性能测试
python benchmarks/compression_benchmark.py
```

更多测试选项和参数说明请参考 `docs/QUICKSTART.md`。

## 项目状态

✅ **代码框架已完成** - 所有集成代码、测试脚本和文档已创建

⚠️ **需要在GPU环境中测试** - 需要在支持CUDA的GPU环境中编译和测试

📖 **详细文档**：
- `PROJECT_SUMMARY.md` - 项目完成总结
- `docs/DEPLOYMENT.md` - 部署指南
- `docs/INTEGRATION_GUIDE.md` - 集成指南

## 参考文献

1. Huang, Y., et al. (2023). cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework. SC'23.
2. Kwon, W., et al. (2023). Efficient memory management for large language model serving with pagedattention. SOSP'23.

