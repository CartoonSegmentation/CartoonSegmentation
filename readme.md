# CartoonSegmentation

Implementations of the paper _Instance-guided Cartoon Editing with a Large-scale Dataset_, including an instance segmentation for cartoon/anime characters and some visual techniques built around it.


[![arXiv](https://img.shields.io/badge/arXiv-2312.01943-<COLOR>)](http://arxiv.org/abs/2312.01943)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CartoonSegmentation/CartoonSegmentation/blob/main/run_in_colab.ipynb)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://cartoonsegmentation.github.io/)

<p float="center">
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser00.jpg" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser01.jpg" width="24%" />
  <img src="https://github.com/CartoonSegmentation/CartoonSegmentation/assets/51270320/10301ee4-09c1-45a9-8672-7e0a3cbd1c20" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser03.jpg" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser10.jpg" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser11.jpg" width="24%" />
  <img src="https://github.com/CartoonSegmentation/CartoonSegmentation/assets/51270320/602f8e5b-bec2-4f07-af50-b72d6411da70" width="24%" />
  <img src="https://cartoonsegmentation.github.io/AnimeIns_files/teaser/teaser13.jpg" width="24%" />
</p>



## Preperation

### Install Dependencies

Install Python 3.10 and pytorch:

```bash
conda create -n anime-seg python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate anime-seg
```

Install mmdet:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
pip install -r requirements.txt
```

### Download models

```bash
huggingface-cli lfs-enable-largefiles .
mkdir models
git clone https://huggingface.co/dreMaz/AnimeInstanceSegmentation models/AnimeInstanceSegmentation


```

## Run Segmentation

See `run_segmentation.ipynb``. 

Besides, we have prepared a simple [Huggingface Space](https://huggingface.co/spaces/ljsabc/AnimeIns_CPU) for you to test with the segmentation on the browser. 

![A workable demo](https://animeins.oss-cn-shenzhen.aliyuncs.com/imas.jpg)
*Copyright BANDAI NAMCO Entertainment Inc., We believe this is a fair use for research and educational purpose only.*


## Run 3d Kenburns


https://github.com/dmMaze/CartoonSegmentation/assets/51270320/503c87c3-39d7-40f8-88f9-3ead20e1e5c5



Install cupy following https://docs.cupy.dev/en/stable/install.html  

Run
``` python
python run_kenburns.py --cfg configs/3dkenburns.yaml --input-img examples/kenburns_lion.png
```
or with the interactive interface:
``` python
python naive_interface.py --cfg configs/3dkenburns.yaml
```
and open http://localhost:8080 in your browser.

Please read configs/3dkenburns.yaml for more advanced settings.  

To use Marigold as depth estimator, run
```
git submodule update --init --recursive
```
and set ```depth_est``` to ```marigold``` in configs/3dkenburns.yaml


### Better Inpainting using Stable-diffusion

To get better inpainting results with Stable-diffusion, you need to install stable-diffusion-webui first, and download the tagger: 
``` bash
git clone https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2 models/wd-v1-4-swinv2-tagger-v2
```

If you're on Windows, download compiled libs from https://github.com/AnimeIns/PyPatchMatch/releases/tag/v1.0 and save them to data/libs, otherwise, you need to compile patchmatch in order to run 3dkenburns or style editing:

#### Compile Patchmatch

``` bash
mkdir -P data/libs
apt install build-essential libopencv-dev -y
git clone https://github.com/AnimeIns/PyPatchMatch && cd PyPatchMatch

mkdir release && cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
make

cd ../..
mv PyPatchMatch/release/libpatchmatch_inpaint.so ./data/libs
rm -rf PyPatchMatch
```
<i>If you have activated conda and encountered `GLIBCXX_3.4.30' not found or libpatchmatch_inpaint.so: cannot open shared object file: No such file or directory, follow the solution here https://askubuntu.com/a/1445330 </i>

Launch the stable-diffusion-webui with argument `--api` and set the base model to `sd-v1-5-inpainting`, modify `inpaint_type: default` to `inpaint_type: ldm` in configs/3dkenburns.yaml.   

Finally, run 3dkenburns with pre-mentioned commands.


## Run Style Editing
It also requires stable-diffusion-webui, patchmatch, and the danbooru tagger, so please follow the `Run 3d Kenburns` and download/install these first.  
Download [sd_xl_base_1.0_0.9vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors), style [lora](https://civitai.com/models/124347/xlmoreart-full-xlreal-enhancer) and [diffusers_xl_canny_mid](https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors) and save them to corresponding directory in stable-diffusion-webui, launch stable-diffusion-webui with argument `--argment` and set `sd_xl_base_1.0_0.9vae` as base model, then run

```
python run_style.py --img_path examples/kenburns_lion.png --cfg configs/3d_pixar.yaml
```
set `onebyone` to False in configs/3d_pixar.yaml to disable instance-aware style editing.


## Run Web UI (Including both _3D Ken Burns_ and _Style Editing_), based on Gradio
All required libraries and configurations have been included, now we just need to execute the Web UI from its Launcher: 

```
python Web_UI/Launcher.py
```
In default configurations, you can find the Web UI here:
- http://localhost:1234 in local
- A random temporary public URL generated by Gradio, such like this: https://1ec9f82dc15633683e.gradio.live