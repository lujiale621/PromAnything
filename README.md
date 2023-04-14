
### Installation





### Installation
Requires `python>=3.8`
```bash
conda create -n myenv python==3.10.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e segment_anything
pip install -r lama/requirements.txt 
conda install scikit-image
conda install scikit-learn
pip install pytorch-lightning==1.2.9
pip install albumentations
pip install omegaconf
pip install fastapi
pip install gradio
pip install uvicorn
pip install pynvml
pip install flask
cd GroundingDINO pip install -e .
pip install imgaug
pip install diffusers
pip install accelerate

hugface下载模型
你可以在conda环境中使用以下命令来安装git lfs：

```
conda install -c conda-forge git-lfs
```
git lfs install
这将从conda-forge channel安装git lfs。 安装完成后，可以通过运行以下命令来验证是否成功安装了git lfs：

```
git lfs version
```



nohup CUDA_VISIBLE_DEVICES=3 python app/app_gradio.py &>/dev/null &

```



## Acknowledgments
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [LaMa](https://github.com/advimman/lama)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)








