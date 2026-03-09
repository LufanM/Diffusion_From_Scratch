# 简介

本项目用于从0构建diffusion model，包含三部分，DDPM(位于[diffusion_model文件](./diffusion_model)下)，stable diffusion model（stable_diffusion_model文件夹下），flow matching（flow_matching文件夹下）



## 运行环境
创建几乎纯净的conda环境
```bash
conda create -n ddpm_study python=3.9.23 -y
```

```bash
pip3 install -r requirement.txt
```
如果用pip安装过慢，用conda安装
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch 
```

拉取子仓
```bash
git submodule update --init --recursive
```



## DDPM

![image-20260306194124690](./README.assets/image-20260306194124690.png)

* 根据数学原理的推导代码在[diffusion_sample.py](./diffusion_model/diffusion_sample.py)中
* 然后为了学习训练过程，调整了diffusion_sample.py 为diffusion_image.py，为了方便图像处理，还加入了如下几个模块
  * unet.py模块取代MLP
  * train_image.py用于训练
  * infer_image.py用于推理



## Stable Diffusion Model

视频课程:https://www.bilibili.com/video/BV1Mm4y117Ci  (stable - diffusion model是DDPM的基础上引入VAE对像素进行压缩，加速训练)

**主要不同于DDPM的地方**

* 训练的模型能文生图
* 相比DDPM额外引入了**CLIPTextModel**编码模块（用于输入文本-图像信息），**VAE**模块（用于图像压缩）

主要逻辑图如下：

![image-20260306194253002](./README.assets/image-20260306194253002.png)





## Flow Matching 
