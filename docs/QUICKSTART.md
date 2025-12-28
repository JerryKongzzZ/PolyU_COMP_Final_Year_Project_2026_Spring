# 快速开始指南

## 概述

本指南帮助您快速开始使用cuSZp集成到vLLM的项目。

## 当前状态

✅ **已完成**：
- 所有代码框架（C++包装器、Python集成、测试脚本）
- 完整的文档（部署指南、集成指南）
- 构建脚本和Docker配置

⚠️ **需要在GPU环境中完成**：
- 编译cuSZp和包装器
- 运行测试和性能评估
- 集成到vLLM并测试

## 开发工作

### 1. 代码审查和编辑

所有代码文件都可以编辑和审查：

```bash
# 查看项目结构
tree -L 3

# 编辑代码
code integration/cuszp_wrapper/cuszp_wrapper.cpp
code integration/compression_pipeline/compressed_swap.py
```

### 2. 版本控制

```bash
# 提交代码
git add .
git commit -m "Add integration code"
git push
```

### 3. 文档阅读

- `README.md` - 项目概述
- `PROJECT_SUMMARY.md` - 完成总结
- `docs/DEPLOYMENT.md` - 部署方案
- `docs/INTEGRATION_GUIDE.md` - 集成步骤

## 在GPU环境中需要做什么

### 步骤1: 准备GPU环境

选择以下方案之一：

**方案A: 云GPU（推荐）**
- AWS EC2 (g4dn.xlarge或更高)
- Google Cloud Platform
- Azure GPU VM

**方案B: Docker容器**

详细Docker使用说明请参考 `docs/DEPLOYMENT.md` 中的"Docker容器"部分。

快速开始：
```bash
cd docker
chmod +x run.sh
./run.sh build    # 构建镜像
./run.sh run      # 运行容器
```

**方案C: 远程Linux服务器**
- SSH到具有NVIDIA GPU的服务器

### 步骤2: 克隆项目

```bash
git clone <your-repo-url>
cd FYP
```

### 步骤3: 构建项目

**推荐：使用自动化脚本**

```bash
chmod +x build.sh
./build.sh
```

该脚本会自动完成所有编译步骤。

**手动构建**（如需自定义选项）：

详细手动构建步骤请参考 `docs/INTEGRATION_GUIDE.md` 中的"构建步骤"部分。

### 步骤4: 运行测试

```bash
# 基线性能测试
python benchmarks/baseline_profiling.py \
    --device-id 0 \
    --output baseline_results.json

# 压缩性能测试
python benchmarks/compression_benchmark.py \
    --device-id 0 \
    --tensor-size 1048576 \
    --error-bound 1e-4 \
    --output compression_results.json
```

### 步骤5: 集成到vLLM

参考 `docs/INTEGRATION_GUIDE.md` 中的详细步骤。

简要流程：
1. 克隆vLLM源代码
2. 复制集成代码到vLLM
3. 修改vLLM的`platforms/cuda.py`
4. 编译和测试

## 常见问题

### Q: 为什么需要GPU环境？

A: 本项目需要NVIDIA GPU和CUDA支持，必须在Linux + NVIDIA GPU环境中运行。

### Q: 如何获取GPU环境？

A: 有几种选择：
1. 使用云GPU服务（AWS/GCP/Azure）
2. 使用学校的GPU服务器
3. 使用Docker（需要Linux环境）

### Q: 代码已经完成了吗？

A: 代码框架已完成，但需要在GPU环境中：
- 编译和测试
- 调试和优化
- 性能评估

### Q: 如何开始？

A: 
1. 阅读文档，理解代码结构
2. 在GPU环境中：按照本指南的步骤操作

## 下一步

1. **阅读文档**：理解项目架构和设计
2. **准备GPU环境**：选择云服务或使用现有服务器
3. **构建和测试**：在GPU环境中编译和运行
4. **集成和评估**：集成到vLLM并评估性能

## 获取帮助

- 查看 `docs/INTEGRATION_GUIDE.md` 了解集成细节
- 查看 `docs/DEPLOYMENT.md` 了解部署选项
- 查看 `PROJECT_SUMMARY.md` 了解项目完成情况

