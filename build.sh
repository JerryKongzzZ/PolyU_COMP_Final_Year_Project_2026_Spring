#!/bin/bash
# 构建脚本：编译cuSZp和集成代码

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Building cuSZp and Integration Code"
echo "=========================================="

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Please install CUDA Toolkit."
    exit 1
fi

echo "CUDA Version: $(nvcc --version | grep release)"

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.21+."
    exit 1
fi

echo "CMake Version: $(cmake --version | head -n 1)"

# 步骤1: 编译cuSZp
echo ""
echo "Step 1: Building cuSZp..."
cd cuSZp

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
make -j$(nproc)
make install

echo "cuSZp built successfully!"

# 步骤2: 编译cuSZp包装器
echo ""
echo "Step 2: Building cuSZp wrapper..."
cd ../../integration/cuszp_wrapper

if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# 检查pybind11
python3 -c "import pybind11" 2>/dev/null || {
    echo "Installing pybind11..."
    pip3 install pybind11
}

cmake ..
make -j$(nproc)

echo "cuSZp wrapper built successfully!"

# 步骤3: 验证构建
echo ""
echo "Step 3: Verifying build..."

if [ -f "cuszp_wrapper_cpp.so" ] || [ -f "cuszp_wrapper_cpp.cpython*.so" ]; then
    echo "✓ cuSZp wrapper library found"
else
    echo "✗ cuSZp wrapper library not found"
    exit 1
fi

if [ -d "../../../cuSZp/install/lib" ]; then
    echo "✓ cuSZp library installed"
else
    echo "✗ cuSZp library not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set PYTHONPATH to include integration/compression_pipeline:"
echo "   export PYTHONPATH=\$PYTHONPATH:$(pwd)/../../integration/compression_pipeline"
echo ""
echo "2. Test the integration:"
echo "   python3 benchmarks/compression_benchmark.py"
echo ""

