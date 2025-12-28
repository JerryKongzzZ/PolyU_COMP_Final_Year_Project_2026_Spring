# vLLM集成指南

## 概述

本指南详细说明如何将cuSZp压缩功能集成到vLLM框架中。

## 前提条件

1. **硬件要求**
   - NVIDIA GPU（支持CUDA 11.0+）
   - 足够的GPU内存和CPU内存

2. **软件要求**
   - Linux操作系统（Ubuntu 20.04+）
   - CUDA Toolkit 11.8+
   - Python 3.9+
   - CMake 3.21+
   - vLLM源代码（需要修改版本）

## 构建步骤

### 1. 编译cuSZp

```bash
cd cuSZp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
make -j$(nproc)
make install
```

### 2. 编译cuSZp包装器

```bash
cd ../../integration/cuszp_wrapper
mkdir build && cd build
cmake ..
make -j$(nproc)
```

或者使用提供的构建脚本：

```bash
chmod +x build.sh
./build.sh
```

### 3. 设置Python路径

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/integration/compression_pipeline
```

## 集成到vLLM

### 方法1: 修改vLLM源代码（推荐用于开发）

1. **克隆vLLM源代码**

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

2. **复制集成代码**

```bash
# 复制压缩管道代码
cp -r /path/to/FYP/integration/compression_pipeline vllm/integration/

# 复制编译好的包装器库
cp /path/to/FYP/integration/cuszp_wrapper/build/cuszp_wrapper_cpp*.so \
   vllm/integration/compression_pipeline/
```

3. **修改vLLM的platforms/cuda.py**

在文件开头添加导入：

```python
try:
    from vllm.integration.compression_pipeline.compressed_swap import (
        get_compressed_swap_manager,
        initialize_compressed_swap
    )
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False
    logger.warning("Compression pipeline not available")
```

修改`swap_out_blocks_to_host`方法：

```python
@classmethod
def swap_out_blocks_to_host(
    cls,
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    src_block_indices: torch.Tensor,
    dst_block_indices: torch.Tensor,
) -> None:
    """Copy blocks from GPU to host (CPU)."""
    manager = get_compressed_swap_manager()
    if manager and manager.enable_compression:
        manager.swap_out_blocks_to_host_compressed(
            src_cache, dst_cache, src_block_indices, dst_block_indices
        )
    else:
        # 原始实现
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()
```

类似地修改`swap_in_blocks_from_host`方法（如果存在）。

4. **在vLLM初始化时启用压缩**

在vLLM的初始化代码中（例如`vllm/engine/llm_engine.py`），添加：

```python
from vllm.integration.compression_pipeline.compressed_swap import (
    initialize_compressed_swap
)

# 在LLMEngine初始化时
if COMPRESSION_AVAILABLE:
    initialize_compressed_swap(
        enable_compression=True,
        error_bound=1e-4,  # 根据需求调整
        use_relative_error=True,
        encoding_mode="plain",
        device_id=0
    )
```

### 方法2: 使用环境变量（推荐用于生产）

可以通过环境变量控制压缩功能，无需修改vLLM源代码：

```bash
export VLLM_ENABLE_COMPRESSION=1
export VLLM_COMPRESSION_ERROR_BOUND=1e-4
export VLLM_COMPRESSION_MODE=plain
```

然后在vLLM代码中读取这些环境变量。

## 配置参数

### 压缩配置

- `enable_compression`: 是否启用压缩（默认：True）
- `error_bound`: 错误边界（默认：1e-4）
  - 较小的值：更高的精度，较低的压缩比
  - 较大的值：较低的精度，较高的压缩比
- `use_relative_error`: 是否使用相对错误边界（默认：True）
- `encoding_mode`: 编码模式
  - `"fixed"`: 适用于非结构化数据（如随机权重）
  - `"plain"`: 适用于局部平滑数据（推荐用于KV cache）
  - `"outlier"`: 适用于全局平滑数据（最高压缩比）

### 性能调优建议

1. **错误边界选择**
   - 对于KV cache，建议从1e-4开始测试
   - 根据模型精度要求调整

2. **编码模式选择**
   - KV cache通常具有局部相关性，建议使用`"plain"`模式
   - 如果压缩比不够，可以尝试`"outlier"`模式

3. **异步执行**
   - 确保使用CUDA流来实现压缩/解压缩与数据传输的重叠
   - 这可以显著提高性能

## 测试和验证

### 1. 运行基线性能测试

```bash
python benchmarks/baseline_profiling.py \
    --device-id 0 \
    --tensor-sizes 1024 4096 16384 65536 262144 1048576 \
    --iterations 100 \
    --output baseline_results.json
```

### 2. 运行压缩性能测试

```bash
python benchmarks/compression_benchmark.py \
    --device-id 0 \
    --tensor-size 1048576 \
    --error-bound 1e-4 \
    --encoding-mode plain \
    --iterations 100 \
    --output compression_results.json
```

### 3. 运行端到端测试

使用vLLM运行一个简单的推理任务，比较启用和禁用压缩的性能：

```bash
# 禁用压缩
python -m vllm.entrypoints.openai.api_server \
    --model <model_name> \
    --disable-compression

# 启用压缩
python -m vllm.entrypoints.openai.api_server \
    --model <model_name> \
    --enable-compression \
    --compression-error-bound 1e-4
```

## 故障排除

### 问题1: cuSZp包装器导入失败

**错误**: `ImportError: No module named 'cuszp_wrapper_cpp'`

**解决方案**:
1. 确保已编译包装器库
2. 检查Python路径设置
3. 确保库文件在正确的位置

### 问题2: CUDA错误

**错误**: `CUDA error: out of memory` 或类似的CUDA错误

**解决方案**:
1. 检查GPU内存是否足够
2. 减少批次大小
3. 检查CUDA版本兼容性

### 问题3: 压缩失败

**错误**: 压缩操作返回失败

**解决方案**:
1. 检查错误边界是否合理
2. 尝试不同的编码模式
3. 检查输入数据格式

## 性能分析

### 使用Nsight Systems

```bash
nsys profile --trace=cuda,nvtx \
    python -m vllm.entrypoints.openai.api_server \
    --model <model_name> \
    --enable-compression
```

### 使用Nsight Compute

```bash
ncu --set full \
    python -m vllm.entrypoints.openai.api_server \
    --model <model_name> \
    --enable-compression
```

## 下一步

1. **优化内存管理**: 实现更高效的压缩缓冲区管理
2. **动态调整**: 根据工作负载动态调整压缩参数
3. **多GPU支持**: 扩展支持多GPU环境
4. **性能监控**: 添加详细的性能指标收集

