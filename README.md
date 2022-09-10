# DPIR
Plug-and-Play Image Restoration with Deep Denoiser Prior, based on https://github.com/cszn/DPIR.


## Dependencies
- [ncnn](https://github.com/Tencent/ncnn). Note that the official [Python package](https://pypi.org/project/ncnn/) is not built with Vulkan compute support. Windows users can install the wheel from https://github.com/HolyWu/ncnn/releases/latest. Non-windows users have to build from source using my fork (requires Vulkan SDK):
```
git clone https://github.com/HolyWu/ncnn.git --depth 1
cd ncnn
git submodule update --init --recursive --depth 1
pip install .
```
- [VapourSynth](http://www.vapoursynth.com/) R55 or later.


## Installation
```
pip install -U vsdpir_ncnn
python -m vsdpir_ncnn
```


## Usage
```python
from vsdpir_ncnn import dpir

ret = dpir(clip)
```

See `__init__.py` for the description of the parameters.
