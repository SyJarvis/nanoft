# 安装

## 环境要求

- Python 3.10 或更高版本。
- PyTorch 2.8 或更高版本。
- Transformers 4.51 或更高版本。

NanoFT 的核心依赖还包括 `datasets` 与 `safetensors`。准确版本范围以仓库的
`pyproject.toml` 为准。

## 从源码安装

NanoFT 当前推荐从源码以 editable 模式安装：

```bash
git clone <nanoft-repository-url>
cd nanoft
pip install -e .
```

确认当前 Python 加载的是刚安装的仓库：

```bash
python -c "import nanoft; print(nanoft.__version__, nanoft.__file__)"
```

如果路径指向另一份工作区或旧环境，请先卸载旧包，或在正确的虚拟环境中
重新执行 `pip install -e .`。

## 安装训练框架

NanoFT 不包含训练循环。运行仓库里的 TRL 示例时，安装训练扩展：

```bash
pip install -e '.[trl]'
```

NanoFT 核心包不锁定 TRL、Accelerate 或 PEFT 的具体版本。请根据当前
PyTorch 与 Transformers 环境选择相互兼容的版本。

PEFT 仅在需要用 `PeftModel` 直接加载兼容 adapter 时安装：

```bash
pip install -e '.[peft]'
```

QLoRA 需要 CUDA 和 bitsandbytes。NanoFT 提供对应的配置、量化包装和
反量化合并 API，但当前仍是实验性能力：

```bash
pip install -e '.[qlora]'
```

## 设备说明

- CUDA：优先使用 BF16；不支持 BF16 的 GPU 使用 FP16。
- Apple MPS：默认使用 FP32，也可以显式选择 FP16 降低内存占用。
- CPU：适合 API 验证和小模型测试，不适合大型模型 SFT。

NanoFT 提供 `detect_device()` 和 `get_recommended_dtype()`，但最终 dtype、
device map 和混合精度仍由调用方及训练框架控制。

完整策略见 [CUDA 与 MPS 支持](device-support.md)。
