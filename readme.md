# Preperation

## Install Dependencies

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

## Download models

Download models：

```bash
huggingface-cli lfs-enable-largefiles .
mkdir models
git clone https://huggingface.co/motionsomething/AnimeInstanceSegmentation models/AnimeInstanceSegmentation


```

# Run Segmentation

See run_segmentation.ipynb

# Run 3d Kenburns


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


## Better Inpainting using Stable-diffusion

To get better inpainting results with Stable-diffusion, you need to install stable-diffusion-webui first, and download the tagger: 
``` bash
git clone https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2 models/wd-v1-4-swinv2-tagger-v2
```

If you're on Windows, download compiled libs from https://github.com/AnimeIns/PyPatchMatch/releases/tag/v1.0 and save them to data/libs, otherwise, you need to compile patchmatch in order to run 3dkenburns or style editing:

### Compile Patchmatch

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


# Run Style Editing
It also requires stable-diffusion-webui, patchmatch, and the danbooru tagger, so please follow the `Run 3d Kenburns` and download/install these first.  
Download [sd_xl_base_1.0_0.9vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors), style [lora](https://civitai.com/models/124347/xlmoreart-full-xlreal-enhancer) and [diffusers_xl_canny_mid](https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_mid.safetensors) and save them to corresponding directory in stable-diffusion-webui, launch stable-diffusion-webui with argument `--argment` and set `sd_xl_base_1.0_0.9vae` as base model, then run

```
python run_style.py --img_path examples/kenburns_lion.png --cfg configs/3d_pixar.yaml
```
set `onebyone` to False in configs/3d_pixar.yaml to disable instance-aware style editing.
