import enum
import typing
import vapoursynth as vs
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "Backend",
    "BackendV2",
    "Waifu2x",
    "Waifu2xModel",
    "DPIR",
    "DPIRModel",
    "RealESRGAN",
    "RealESRGANModel",
    "RealESRGANv2",
    "RealESRGANv2Model",
    "CUGAN",
    "RIFE",
    "RIFEModel",
    "RIFEMerge",
    "SAFA",
    "SAFAModel",
    "SAFAAdaptiveMode",
    "SCUNet",
    "SCUNetModel",
    "SwinIR",
    "SwinIRModel",
    "ArtCNN",
    "ArtCNNModel",
    "inference",
    "flexible_inference",
]

plugins_path: str
trtexec_path: str
migraphx_driver_path: str
tensorrt_rtx_path: str
models_path: str

class Backend:
    @dataclass(frozen=False)
    class ORT_CPU:
        num_streams: int = ...
        verbosity: int = ...
        fp16: bool = ...
        fp16_blacklist_ops: typing.Sequence[str] | None = ...
        output_format: int = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class ORT_CUDA:
        device_id: int = ...
        cudnn_benchmark: bool = ...
        num_streams: int = ...
        verbosity: int = ...
        fp16: bool = ...
        use_cuda_graph: bool = ...
        fp16_blacklist_ops: typing.Sequence[str] | None = ...
        prefer_nhwc: bool = ...
        output_format: int = ...
        tf32: bool = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class OV_CPU:
        fp16: bool = ...
        num_streams: int | str = ...
        bind_thread: bool = ...
        fp16_blacklist_ops: typing.Sequence[str] | None = ...
        bf16: bool = ...
        num_threads: int = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class TRT:
        max_shapes: tuple[int, int] | None = ...
        opt_shapes: tuple[int, int] | None = ...
        fp16: bool = ...
        device_id: int = ...
        workspace: int | None = ...
        verbose: bool = ...
        use_cuda_graph: bool = ...
        num_streams: int = ...
        use_cublas: bool = ...
        static_shape: bool = ...
        tf32: bool = ...
        log: bool = ...
        use_cudnn: bool = ...
        use_edge_mask_convolutions: bool = ...
        use_jit_convolutions: bool = ...
        heuristic: bool = ...
        output_format: int = ...
        min_shapes: tuple[int, int] = ...
        faster_dynamic_shapes: bool = ...
        force_fp16: bool = ...
        builder_optimization_level: int = ...
        max_aux_streams: int | None = ...
        short_path: bool | None = ...
        bf16: bool = ...
        custom_env: dict[str, str] = ...
        custom_args: list[str] = ...
        engine_folder: str | None = ...
        max_tactics: int | None = ...
        tiling_optimization_level: int = ...
        l2_limit_for_tiling: int = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class OV_GPU:
        fp16: bool = ...
        num_streams: int | str = ...
        device_id: int = ...
        fp16_blacklist_ops: typing.Sequence[str] | None = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class NCNN_VK:
        fp16: bool = ...
        device_id: int = ...
        num_streams: int = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class ORT_DML:
        device_id: int = ...
        num_streams: int = ...
        verbosity: int = ...
        fp16: bool = ...
        fp16_blacklist_ops: typing.Sequence[str] | None = ...
        output_format: int = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class MIGX:
        device_id: int = ...
        fp16: bool = ...
        opt_shapes: tuple[int, int] | None = ...
        fast_math: bool = ...
        exhaustive_tune: bool = ...
        num_streams: int = ...
        short_path: bool | None = ...
        custom_env: dict[str, str] = ...
        custom_args: list[str] = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class OV_NPU:
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class ORT_COREML:
        num_streams: int = ...
        verbosity: int = ...
        fp16: bool = ...
        fp16_blacklist_ops: typing.Sequence[str] | None = ...
        ml_program: int = ...
        output_format: int = ...
        supports_onnx_serialization: bool = ...

    @dataclass(frozen=False)
    class TRT_RTX:
        fp16: bool = ...
        device_id: int = ...
        workspace: int | None = ...
        verbose: bool = ...
        use_cuda_graph: bool = ...
        num_streams: int = ...
        static_shape: bool = ...
        min_shapes: tuple[int, int] = ...
        opt_shapes: tuple[int, int] | None = ...
        max_shapes: tuple[int, int] | None = ...
        use_cudnn: bool = ...
        use_edge_mask_convolutions: bool = ...
        builder_optimization_level: int = ...
        max_aux_streams: int | None = ...
        short_path: bool | None = ...
        custom_env: dict[str, str] = ...
        custom_args: list[str] = ...
        engine_folder: str | None = ...
        max_tactics: int | None = ...
        tiling_optimization_level: int = ...
        l2_limit_for_tiling: int = ...
        supports_onnx_serialization: bool = ...

backendT = (
    Backend.OV_CPU
    | Backend.ORT_CPU
    | Backend.ORT_CUDA
    | Backend.TRT
    | Backend.OV_GPU
    | Backend.NCNN_VK
    | Backend.ORT_DML
    | Backend.MIGX
    | Backend.OV_NPU
    | Backend.ORT_COREML
    | Backend.TRT_RTX
)

class Waifu2xModel(enum.IntEnum):
    anime_style_art = 0
    anime_style_art_rgb = 1
    photo = 2
    upconv_7_anime_style_art_rgb = 3
    upconv_7_photo = 4
    upresnet10 = 5
    cunet = 6
    swin_unet_art = 7
    swin_unet_photo = 8
    swin_unet_photo_v2 = 9
    swin_unet_art_scan = 10

def Waifu2x(
    clip: vs.VideoNode,
    noise: typing.Literal[-1, 0, 1, 2, 3] = -1,
    scale: typing.Literal[1, 2, 4] = 2,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: Waifu2xModel = ...,
    backend: backendT = ...,
    preprocess: bool = True,
) -> vs.VideoNode: ...

class DPIRModel(enum.IntEnum):
    drunet_gray = 0
    drunet_color = 1
    drunet_deblocking_grayscale = 2
    drunet_deblocking_color = 3

def DPIR(
    clip: vs.VideoNode,
    strength: typing.SupportsFloat | vs.VideoNode | None,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: DPIRModel = ...,
    backend: backendT = ...,
) -> vs.VideoNode: ...

class RealESRGANModel(enum.IntEnum):
    animevideo_xsx2 = 0
    animevideo_xsx4 = 1
    animevideov3 = 2
    animejanaiV2L1 = 5005
    animejanaiV2L2 = 5006
    animejanaiV2L3 = 5007
    animejanaiV3_HD_L1 = 5008
    animejanaiV3_HD_L2 = 5009
    animejanaiV3_HD_L3 = 5010
    Ani4Kv2_G6i2_Compact = 7000
    Ani4Kv2_G6i2_UltraCompact = 7001

RealESRGANv2Model = RealESRGANModel

def RealESRGAN(
    clip: vs.VideoNode,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: RealESRGANv2Model = ...,
    backend: backendT = ...,
    scale: float | None = None,
) -> vs.VideoNode: ...

RealESRGANv2 = RealESRGAN

def CUGAN(
    clip: vs.VideoNode,
    noise: typing.Literal[-1, 0, 1, 2, 3] = -1,
    scale: typing.Literal[2, 3, 4] = 2,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    backend: backendT = ...,
    alpha: float = 1.0,
    version: typing.Literal[1, 2] = 1,
    conformance: bool = True,
) -> vs.VideoNode: ...

class RIFEModel(enum.IntEnum):
    v4_0 = 40
    v4_2 = 42
    v4_3 = 43
    v4_4 = 44
    v4_5 = 45
    v4_6 = 46
    v4_7 = 47
    v4_8 = 48
    v4_9 = 49
    v4_10 = 410
    v4_11 = 411
    v4_12 = 412
    v4_12_lite = 4121
    v4_13 = 413
    v4_13_lite = 4131
    v4_14 = 414
    v4_14_lite = 4141
    v4_15 = 415
    v4_15_lite = 4151
    v4_16_lite = 4161
    v4_17 = 417
    v4_17_lite = 4171
    v4_18 = 418
    v4_19 = 419
    v4_20 = 420
    v4_21 = 421
    v4_22 = 422
    v4_22_lite = 4221
    v4_23 = 423
    v4_24 = 424
    v4_25 = 425
    v4_25_lite = 4251
    v4_25_heavy = 4252
    v4_26 = 426
    v4_26_heavy = 4262

def RIFEMerge(
    clipa: vs.VideoNode,
    clipb: vs.VideoNode,
    mask: vs.VideoNode,
    scale: float = 1.0,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: RIFEModel = ...,
    backend: backendT = ...,
    ensemble: bool = False,
    _implementation: typing.Literal[1, 2] | None = None,
) -> vs.VideoNode: ...
def RIFE(
    clip: vs.VideoNode,
    multi: int | Fraction = 2,
    scale: float = 1.0,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: RIFEModel = ...,
    backend: backendT = ...,
    ensemble: bool = False,
    video_player: bool = False,
    _implementation: typing.Literal[1, 2] | None = None,
) -> vs.VideoNode: ...

class SAFAModel(enum.IntEnum):
    v0_1 = 1
    v0_2 = 2
    v0_3 = 3
    v0_4 = 4
    v0_5 = 5

class SAFAAdaptiveMode(enum.IntEnum):
    non_adaptive = 0
    adaptive1x = 1
    adaptive = 2

def SAFA(
    clip: vs.VideoNode,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: SAFAModel = ...,
    adaptive: SAFAAdaptiveMode = ...,
    backend: backendT = ...,
) -> vs.VideoNode: ...

class SCUNetModel(enum.IntEnum):
    scunet_color_15 = 0
    scunet_color_25 = 1
    scunet_color_50 = 2
    scunet_color_real_psnr = 3
    scunet_color_real_gan = 4
    scunet_gray_15 = 5
    scunet_gray_25 = 6
    scunet_gray_50 = 7

def SCUNet(
    clip: vs.VideoNode,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: SCUNetModel = ...,
    backend: backendT = ...,
) -> vs.VideoNode: ...

class SwinIRModel(enum.IntEnum):
    lightweightSR_DIV2K_s64w8_SwinIR_S_x2 = 0
    lightweightSR_DIV2K_s64w8_SwinIR_S_x3 = 1
    lightweightSR_DIV2K_s64w8_SwinIR_S_x4 = 2
    realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_x4_GAN = 3
    realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_x4_PSNR = 5
    classicalSR_DF2K_s64w8_SwinIR_M_x2 = 6
    classicalSR_DF2K_s64w8_SwinIR_M_x3 = 7
    classicalSR_DF2K_s64w8_SwinIR_M_x4 = 8
    classicalSR_DF2K_s64w8_SwinIR_M_x8 = 9
    realSR_BSRGAN_DFO_s64w8_SwinIR_M_x2_GAN = 10
    realSR_BSRGAN_DFO_s64w8_SwinIR_M_x2_PSNR = 11
    realSR_BSRGAN_DFO_s64w8_SwinIR_M_x4_GAN = 12
    realSR_BSRGAN_DFO_s64w8_SwinIR_M_x4_PSNR = 13
    grayDN_DFWB_s128w8_SwinIR_M_noise15 = 14
    grayDN_DFWB_s128w8_SwinIR_M_noise25 = 15
    grayDN_DFWB_s128w8_SwinIR_M_noise50 = 16
    colorDN_DFWB_s128w8_SwinIR_M_noise15 = 17
    colorDN_DFWB_s128w8_SwinIR_M_noise25 = 18
    colorDN_DFWB_s128w8_SwinIR_M_noise50 = 19
    CAR_DFWB_s126w7_SwinIR_M_jpeg10 = 20
    CAR_DFWB_s126w7_SwinIR_M_jpeg20 = 21
    CAR_DFWB_s126w7_SwinIR_M_jpeg30 = 22
    CAR_DFWB_s126w7_SwinIR_M_jpeg40 = 23
    colorCAR_DFWB_s126w7_SwinIR_M_jpeg10 = 24
    colorCAR_DFWB_s126w7_SwinIR_M_jpeg20 = 25
    colorCAR_DFWB_s126w7_SwinIR_M_jpeg30 = 26
    colorCAR_DFWB_s126w7_SwinIR_M_jpeg40 = 27

def SwinIR(
    clip: vs.VideoNode,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: SwinIRModel = ...,
    backend: backendT = ...,
) -> vs.VideoNode: ...

class ArtCNNModel(enum.IntEnum):
    ArtCNN_C4F32 = 0
    ArtCNN_C4F32_DS = 1
    ArtCNN_C16F64 = 2
    ArtCNN_C16F64_DS = 3
    ArtCNN_C4F32_Chroma = 4
    ArtCNN_C16F64_Chroma = 5
    ArtCNN_R16F96 = 6
    ArtCNN_R8F64 = 7
    ArtCNN_R8F64_DS = 8
    ArtCNN_R8F64_Chroma = 9
    ArtCNN_C4F16 = 10
    ArtCNN_C4F16_DS = 11
    ArtCNN_R16F96_Chroma = 12

def ArtCNN(
    clip: vs.VideoNode,
    tiles: int | tuple[int, int] | None = None,
    tilesize: int | tuple[int, int] | None = None,
    overlap: int | tuple[int, int] | None = None,
    model: ArtCNNModel = ...,
    backend: backendT = ...,
) -> vs.VideoNode: ...
def inference(
    clips: vs.VideoNode | list[vs.VideoNode],
    network_path: str,
    overlap: tuple[int, int] = (0, 0),
    tilesize: tuple[int, int] | None = None,
    backend: backendT = ...,
    input_name: str | None = "input",
    batch_size: int = 1,
) -> vs.VideoNode: ...
def flexible_inference(
    clips: vs.VideoNode | list[vs.VideoNode],
    network_path: str,
    overlap: tuple[int, int] = (0, 0),
    tilesize: tuple[int, int] | None = None,
    backend: backendT = ...,
    input_name: str | None = "input",
    flexible_output_prop: str = "vsmlrt_flexible",
    batch_size: int = 1,
) -> list[vs.VideoNode]: ...

class BackendV2:
    @staticmethod
    def TRT(
        *,
        num_streams: int = 1,
        fp16: bool = False,
        tf32: bool = False,
        output_format: int = 0,
        workspace: int | None = None,
        use_cuda_graph: bool = False,
        static_shape: bool = True,
        min_shapes: tuple[int, int] = (0, 0),
        opt_shapes: tuple[int, int] | None = None,
        max_shapes: tuple[int, int] | None = None,
        force_fp16: bool = False,
        use_cublas: bool = False,
        use_cudnn: bool = False,
        device_id: int = 0,
        **kwargs: typing.Any,
    ) -> Backend.TRT: ...
    @staticmethod
    def NCNN_VK(
        *, num_streams: int = 1, fp16: bool = False, device_id: int = 0, **kwargs: typing.Any
    ) -> Backend.NCNN_VK: ...
    @staticmethod
    def ORT_CUDA(
        *,
        num_streams: int = 1,
        fp16: bool = False,
        cudnn_benchmark: bool = True,
        device_id: int = 0,
        **kwargs: typing.Any,
    ) -> Backend.ORT_CUDA: ...
    @staticmethod
    def OV_CPU(
        *,
        num_streams: int | str = 1,
        bf16: bool = False,
        bind_thread: bool = True,
        num_threads: int = 0,
        **kwargs: typing.Any,
    ) -> Backend.OV_CPU: ...
    @staticmethod
    def ORT_CPU(*, num_streams: int = 1, **kwargs: typing.Any) -> Backend.ORT_CPU: ...
    @staticmethod
    def OV_GPU(
        *, num_streams: int | str = 1, fp16: bool = False, device_id: int = 0, **kwargs: typing.Any
    ) -> Backend.OV_GPU: ...
    @staticmethod
    def ORT_DML(
        *, device_id: int = 0, num_streams: int = 1, fp16: bool = False, **kwargs: typing.Any
    ) -> Backend.ORT_DML: ...
    @staticmethod
    def MIGX(
        *, fp16: bool = False, opt_shapes: tuple[int, int] | None = None, **kwargs: typing.Any
    ) -> Backend.MIGX: ...
    @staticmethod
    def OV_NPU(**kwargs: typing.Any) -> Backend.OV_NPU: ...
    @staticmethod
    def ORT_COREML(*, num_streams: int = 1, fp16: bool = False, **kwargs: typing.Any) -> Backend.ORT_COREML: ...
    @staticmethod
    def TRT_RTX(
        *,
        num_streams: int = 1,
        fp16: bool = False,
        workspace: int | None = None,
        use_cuda_graph: bool = False,
        static_shape: bool = True,
        min_shapes: tuple[int, int] = (0, 0),
        opt_shapes: tuple[int, int] | None = None,
        max_shapes: tuple[int, int] | None = None,
        device_id: int = 0,
        **kwargs: typing.Any,
    ) -> Backend.TRT_RTX: ...

def calc_tilesize(
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    width: int,
    height: int,
    multiple: int,
    overlap_w: int,
    overlap_h: int,
) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]: ...
def init_backend(backend: backendT, trt_opt_shapes: typing.Tuple[int, int]) -> backendT: ...
