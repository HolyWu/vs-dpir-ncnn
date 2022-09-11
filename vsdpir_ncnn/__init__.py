from __future__ import annotations

import math
import os.path as osp

import ncnn
import numpy as np
import vapoursynth as vs
from vsutil import fallback

dir_name = osp.dirname(__file__)


class DPIR:
    """Deep Plug-and-Play Image Restoration"""

    def __init__(self, task: str = 'denoise', color: bool = True, gpu_id: int | None = None, fp16: bool = True) -> None:
        """
        :param task:    Task to perform. Must be 'deblock' or 'denoise'.
        :param color:   Color model or gray model. Must match the format of the clip on which to be processed.
        :param gpu_id:  The GPU ID.
        :param fp16:    Enable FP16 mode.
        """
        if not hasattr(ncnn, 'get_gpu_count'):
            raise vs.Error('DPIR: the installed ncnn Python package is not built with Vulkan compute support')

        assert ncnn.get_gpu_count() > 0

        task = task.lower()

        if task not in ['deblock', 'denoise']:
            raise vs.Error("DPIR: task must be 'deblock' or 'denoise'")

        if gpu_id is not None and (gpu_id < 0 or gpu_id >= ncnn.get_gpu_count()):
            raise vs.Error('DPIR: invalid GPU device')

        if osp.getsize(osp.join(dir_name, 'drunet_color.bin')) == 0:
            raise vs.Error("DPIR: model files have not been downloaded. run 'python -m vsdpir_ncnn' first")

        color_or_gray = 'color' if color else 'gray'
        model_name = f'drunet_deblocking_{color_or_gray}' if task == 'deblock' else f'drunet_{color_or_gray}'
        model_path = osp.join(dir_name, model_name)

        self.task = task

        self.net = ncnn.Net()
        self.net.opt.num_threads = 1
        self.net.opt.use_fp16_packed = fp16
        self.net.opt.use_fp16_storage = fp16
        self.net.opt.use_vulkan_compute = True
        self.net.set_vulkan_device(fallback(gpu_id, ncnn.get_default_gpu_index()))
        self.net.load_param(model_path + '.param')
        self.net.load_model(model_path + '.bin')

    def run(
        self,
        clip: vs.VideoNode,
        strength: float | vs.VideoNode | None = None,
        tile_w: int = 0,
        tile_h: int = 0,
        tile_pad: int = 8,
    ) -> vs.VideoNode:
        """
        Compute the predictions.

        :param clip:        Clip to process. Only RGBS and GRAYS formats are supported.
        :param strength:    Strength for deblocking/denoising. Defaults to 50.0 for 'deblock', 5.0 for 'denoise'.
                            Also accepts a GRAY8/GRAYS clip for varying strength.
        :param tile_w:      Tile width. As too large images result in the out of GPU memory issue,
                            so this tile option will first crop input images into tiles, and then process each of them.
                            Finally, they will be merged into one image. 0 denotes for do not use tile.
        :param tile_h:      Tile height.
        :param tile_pad:    The pad size for each tile, to remove border artifacts.
        """
        if not isinstance(clip, vs.VideoNode):
            raise vs.Error('DPIR: this is not a clip')

        if clip.format.id not in [vs.RGBS, vs.GRAYS]:
            raise vs.Error('DPIR: only RGBS and GRAYS formats are supported')

        if isinstance(strength, vs.VideoNode):
            if strength.format.id not in [vs.GRAY8, vs.GRAYS]:
                raise vs.Error('DPIR: strength must be of GRAY8/GRAYS format')

            if strength.width != clip.width or strength.height != clip.height or strength.num_frames != clip.num_frames:
                raise vs.Error('DPIR: strength must have the same dimensions and number of frames as main clip')

        if self.task == 'deblock':
            if isinstance(strength, vs.VideoNode):
                noise_level = strength.std.Expr(expr='x 100 /', format=vs.GRAYS)
            else:
                noise_level = clip.std.BlankClip(format=vs.GRAYS, color=fallback(strength, 50.0) / 100)
            clip = clip.std.Limiter()
        else:
            if isinstance(strength, vs.VideoNode):
                noise_level = strength.std.Expr(expr='x 255 /', format=vs.GRAYS)
            else:
                noise_level = clip.std.BlankClip(format=vs.GRAYS, color=fallback(strength, 5.0) / 255)

        def _dpir(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
            img = frame_to_ndarray(f[0])
            noise_level_map = frame_to_ndarray(f[1])
            img = np.concatenate((img, noise_level_map))

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, tile_w, tile_h, tile_pad, self.net)
            elif img.shape[1] % 8 == 0 and img.shape[2] % 8 == 0:
                ex = self.net.create_extractor()
                ex.input('input', ncnn.Mat(img))
                _, out = ex.extract('output')
                output = np.asarray(out)
            else:
                output = mod_pad(img, 8, self.net)

            return ndarray_to_frame(output, f[0].copy())

        return clip.std.ModifyFrame(clips=[clip, noise_level], selector=_dpir)


def frame_to_ndarray(frame: vs.VideoFrame) -> np.ndarray:
    return np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(img: np.ndarray, tile_w: int, tile_h: int, tile_pad: int, net: ncnn.ncnn.Net) -> np.ndarray:
    channel, height, width = img.shape
    output_shape = (channel - 1, height, width)

    # start with black image
    output = np.zeros_like(img, shape=output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # process tile
            if input_tile.shape[1] % 8 == 0 and input_tile.shape[2] % 8 == 0:
                ex = net.create_extractor()
                ex.input('input', ncnn.Mat(input_tile))
                _, out = ex.extract('output')
                output_tile = np.asarray(out)
            else:
                output_tile = mod_pad(input_tile, 8, net)

            # output tile area on total image
            output_start_x = input_start_x
            output_end_x = input_end_x
            output_start_y = input_start_y
            output_end_y = input_end_y

            # output tile area without padding
            output_start_x_tile = input_start_x - input_start_x_pad
            output_end_x_tile = output_start_x_tile + input_tile_width
            output_start_y_tile = input_start_y - input_start_y_pad
            output_end_y_tile = output_start_y_tile + input_tile_height

            # put tile into output image
            output[:, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output


def mod_pad(img: np.ndarray, modulo: int, net: ncnn.ncnn.Net) -> np.ndarray:
    mod_pad_h, mod_pad_w = 0, 0
    h, w = img.shape[1:]

    if h % modulo != 0:
        mod_pad_h = modulo - h % modulo

    if w % modulo != 0:
        mod_pad_w = modulo - w % modulo

    img = np.pad(img, ((0, 0), (0, mod_pad_h), (0, mod_pad_w)), 'reflect')
    ex = net.create_extractor()
    ex.input('input', ncnn.Mat(img))
    _, out = ex.extract('output')
    output = np.asarray(out)
    return output[:, :h, :w]
