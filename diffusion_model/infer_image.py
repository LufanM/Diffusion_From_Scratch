import torch
import torchvision
from unet import UNet
from diffusion_image import Diffusion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
unet = UNet(in_channels=3, base_dim=64, dim_mults=(1,2,4,8), time_dim=256).to(DEVICE)
unet.load_state_dict(torch.load('unet_epoch100.pth'))
unet.eval()

diffusion = Diffusion(model=unet, T=1000, device=DEVICE).to(DEVICE)

# 生成 16 张图片
samples = diffusion.sample(shape=(16, 3, 32, 32))  # [16, 3, 32, 32]

# 反归一化 [-1,1] → [0,1]
samples = (samples + 1) / 2
samples = samples.clamp(0, 1)

# 保存
torchvision.utils.save_image(samples, 'generated.png', nrow=4)
print("图片已保存到 generated.png")