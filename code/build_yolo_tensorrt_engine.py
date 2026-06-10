import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("pretrained_models/YOLO/yolov8n.onnx"),
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("pretrained_models/YOLO/yolov8n.engine"),
        help="Output TensorRT engine path.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=0.5,
        help="TensorRT workspace size in GiB.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 if the platform supports it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = args.onnx.resolve()
    engine_path = args.engine.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
    if not parser.parse(onnx_path.read_bytes()):
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    workspace_bytes = int(float(args.workspace) * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if args.half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled.")
    elif args.half:
        print("FP16 requested, but this platform does not report fast FP16 support.")

    print(f"Building engine with workspace={args.workspace} GiB...")
    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT failed to build a serialized engine.")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(serialized_engine))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT failed to build an engine.")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with engine_path.open("wb") as f:
            f.write(engine.serialize())

    print("===== TensorRT Engine Build Finished =====")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")


if __name__ == "__main__":
    main()
