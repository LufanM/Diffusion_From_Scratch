import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from unet import UNet
from diffusion_image import Diffusion
import os

SAVE_DIR = './checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)  # ✅ 自动创建目录

# ========== 配置 ==========
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE  = 32
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 2e-4
NUM_STEPS   = 1000   # ✅ 改名，避免和 T (transforms) 冲突


# ========== 数据集（CIFAR-10）==========
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # ✅ RGB 3 通道各自归一化
])

dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# ========== 模型 ==========
unet = UNet(
    in_channels=3,
    base_dim=64,
    dim_mults=(1, 2, 4, 8),
    time_dim=256,
).to(DEVICE)

diffusion = Diffusion(
    model=unet,
    T=NUM_STEPS,   # ✅ 用新名字
    device=DEVICE
).to(DEVICE)

optimizer = torch.optim.AdamW(unet.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ========== 训练循环 ==========
def train():
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        step_num = 0

        for images, _ in loader:
            images = images.to(DEVICE)

            loss = diffusion.loss(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step_num += 1
            if step_num % 10 == 0:
                print(f"Epoch {epoch+1:3d} Step {step_num:4d} | Loss: {loss.item():.6f}")

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f'unet_epoch{epoch+1}.pth')
            torch.save(unet.state_dict(), save_path)
            print(f"模型已保存到 {save_path}")


if __name__ == '__main__':
    train()