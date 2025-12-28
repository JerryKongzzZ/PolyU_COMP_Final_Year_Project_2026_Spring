# 压缩传输管道集成说明

## 概述

这个模块实现了在vLLM的CPU-GPU交换中使用cuSZp压缩的功能。

## 架构设计

### 1. cuSZp包装器 (cuszp_wrapper)

C++包装器，提供Python接口来调用cuSZp压缩库。

**主要功能：**
- 压缩GPU张量（用于D2H传输）
- 解压缩GPU张量（用于H2D传输）
- 异步执行支持（CUDA流）
- 内存管理

### 2. 压缩交换管理器 (CompressedSwapManager)

Python类，管理压缩的CPU-GPU交换操作。

**主要功能：**
- 替换vLLM的`swap_out_blocks_to_host`函数
- 替换vLLM的`swap_in_blocks_from_host`函数
- 管理压缩配置
- 错误处理和回退机制

## 工作流程

### D2H (Device-to-Host) 压缩传输

```
GPU KV Cache Block
    ↓
[压缩] (在GPU上，异步)
    ↓
压缩数据
    ↓
[传输] (PCIe，数据量减少)
    ↓
CPU内存 (存储压缩数据)
```

### H2D (Host-to-Device) 解压缩传输

```
CPU内存 (压缩数据)
    ↓
[传输] (PCIe，数据量减少)
    ↓
GPU内存 (压缩数据)
    ↓
[解压缩] (在GPU上，异步)
    ↓
GPU KV Cache Block
```

## 集成步骤

### 1. 编译cuSZp包装器

```bash
cd integration/cuszp_wrapper
mkdir build && cd build
cmake ..
make
```

### 2. 修改vLLM代码

需要在vLLM的`platforms/cuda.py`中修改：

```python
from vllm.integration.compression_pipeline.compressed_swap import (
    get_compressed_swap_manager,
    initialize_compressed_swap
)

# 在CudaPlatform类中修改swap_out_blocks_to_host方法
@classmethod
def swap_out_blocks_to_host(
    cls,
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    src_block_indices: torch.Tensor,
    dst_block_indices: torch.Tensor,
) -> None:
    manager = get_compressed_swap_manager()
    if manager:
        manager.swap_out_blocks_to_host_compressed(
            src_cache, dst_cache, src_block_indices, dst_block_indices
        )
    else:
        # 原始实现
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()
```

### 3. 初始化压缩管理器

在vLLM初始化时调用：

```python
from vllm.integration.compression_pipeline.compressed_swap import (
    initialize_compressed_swap
)

# 在vLLM启动时
initialize_compressed_swap(
    enable_compression=True,
    error_bound=1e-4,
    use_relative_error=True,
    encoding_mode="plain"
)
```

## 配置参数

- `enable_compression`: 是否启用压缩（默认：True）
- `error_bound`: 错误边界（默认：1e-4）
- `use_relative_error`: 是否使用相对错误边界（默认：True）
- `encoding_mode`: 编码模式
  - `"fixed"`: 无delta编码，适用于非结构化数据
  - `"plain"`: 带delta编码，适用于局部平滑数据
  - `"outlier"`: 带delta和异常值保留，适用于全局平滑数据（最高压缩比）

## 注意事项

1. **内存管理**：压缩数据的大小可能小于原始数据，需要修改vLLM的CPU缓存结构来存储压缩数据和元数据。

2. **元数据存储**：需要存储压缩相关的元数据（原始大小、压缩大小、错误边界等）以便正确解压缩。

3. **异步执行**：使用CUDA流来实现压缩/解压缩与数据传输的重叠，提高性能。

4. **错误处理**：如果压缩/解压缩失败，应该回退到原始的非压缩实现。

5. **兼容性**：需要确保与vLLM的其他功能（如PagedAttention、多GPU等）兼容。

## 待实现功能

- [ ] 修改vLLM的CPU缓存结构以支持压缩数据
- [ ] 实现压缩元数据的存储和检索
- [ ] 优化内存池管理
- [ ] 添加性能监控和统计
- [ ] 支持动态调整压缩参数

