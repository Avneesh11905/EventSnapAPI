import onnxruntime as ort


def get_providers(device: str) -> list:
    """Get ONNX Runtime execution providers based on device."""
    if device == "cuda":
        available = ort.get_available_providers()
        providers: list[str | tuple[str, dict[str, object]]] = []
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "DEFAULT",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        if providers:
            providers.append("CPUExecutionProvider")
            return providers
    return ["CPUExecutionProvider"]
