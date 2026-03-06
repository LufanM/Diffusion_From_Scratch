# flowmatching_train.py
import os
import math
from datetime import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

"""
    主要用于对比学习flow mathching 这里是无条件的版本，后续可以改成条件版本
"""


# ---------------- User config (adjusted) ----------------
data_root = "./faces/img_align_celeba/img_align_celeba"  # 数据可以从 https://www.kaggle.com/datasets/jessicali9530/celeba-dataset 下载 CelebA 数据集，并解压到这个路径
save_dir = "./flowmatch_checkpoints"
os.makedirs(save_dir, exist_ok=True)
batch_size = 8
lr = 1e-4
num_epochs = 100
image_size = 104
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4
pin_memory = True
sample_every = 1
num_sample_images = 9
base_ch = 64
# flow integration steps during sampling
flow_steps = 200

# ---------------- Data ----------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),  # [0,1]
])

class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # scale to [-1,1]
        img = img * 2.0 - 1.0
        return img

dataset = CelebADataset(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)

# -- begin model definitions --
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) floats (works for continuous t too)
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)
        h = self.conv1(x)
        t_emb = self.time_emb_proj(t_emb)
        h = h + t_emb[:, :, None, None]
        h = self.conv2(h)
        return h + residual

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        skips = []
        for block in self.blocks:
            x = block(x, t_emb)
            skips.append(x)
        x = self.attn(x)
        x = self.downsample(x)
        return x, skips

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        x = self.upsample(x)
        for block in self.blocks:
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        return x

class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attn(x)
        return x

class EnhancedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, use_attention=True)

        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False)

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        # t: (B,) floats in [0,1] (we will scale if needed)
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        x, s1 = self.down1(x, t_emb); skips.extend(s1)
        x, s2 = self.down2(x, t_emb); skips.extend(s2)
        x, s3 = self.down3(x, t_emb); skips.extend(s3)
        x, s4 = self.down4(x, t_emb); skips.extend(s4)

        x = self.mid(x, t_emb)

        x = self.up4(x, skips, t_emb)
        x = self.up3(x, skips, t_emb)
        x = self.up2(x, skips, t_emb)
        x = self.up1(x, skips, t_emb)

        return self.final(x)

# -- end model definitions --

# ---------------- prepare model, optimizer, scaler ----------------
model = EnhancedUNet(in_ch=3, base_ch=base_ch, time_emb_dim=512, num_res_blocks=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

mse = nn.MSELoss()

# ---------------- helper: save sample grid ----------------
def save_samples(x, epoch):
    # x: tensor in [-1,1], shape (N,3,H,W)
    out = (x.clamp(-1,1) + 1.0) / 2.0  # to [0,1]
    grid = utils.make_grid(out, nrow=int(math.sqrt(out.shape[0]) + 0.999), padding=2)
    filename = os.path.join(save_dir, f"sample_epoch_{epoch:03d}.png")
    utils.save_image(grid, filename)
    print(f"[saved] {filename}")

# ---------------- sampling function (Euler integration) ----------------
@torch.no_grad()
def sample_flow(model, n_samples=8, steps=200, device=device):
    model.eval()
    # start from noise x ~ N(0,1)
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    # integrate from t=0..1
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples,), float(i) / steps, device=device, dtype=torch.float32)  # t in [0,1)
        u = model(x, t)  # predicted vector field
        x = x + u * dt
    model.train()
    return x.clamp(-1,1)

# ---------------- training loop ----------------
print("Starting training... device:", device)
total_loss = 0.0
step_count = 0
global_step=0
for epoch in range(num_epochs):
    for z in loader:
        z = z.to(device)  # x (B,3,H,W) in [-1,1]
        B = z.shape[0]

        # sample noise x_0 ~ N(0,1)
        x_0 = torch.randn_like(z)

        # sample t ~ Uniform(0,1)
        t = torch.rand(B, device=device, dtype=torch.float32)

        # construct x_t = t * z + (1 - t) * x_0
        t_broadcast = t.view(B, 1, 1, 1)
        x_t = t_broadcast * z + (1.0 - t_broadcast) * x_0

        # target vector field u_target = z - x_0 (dx_t/dt)
        u_target = (z - x_0).detach()

        optimizer.zero_grad()

        pred = model(x_t, t)  # t shape (B,)
        loss = mse(pred, u_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        total_loss += loss.item()
        step_count += 1
        if global_step % 10 == 0:
            avg_loss = total_loss / step_count
            print(f"Epoch {epoch:03d} Step {global_step:06d} Average Loss: {avg_loss:.6f}")
            total_loss = 0.0
            step_count = 0

        global_step += 1
    if epoch % sample_every == 0:
        samples = sample_flow(model, n_samples=num_sample_images, steps=flow_steps, device=device)
        save_samples(samples, epoch)
    # end epoch: save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch
    }
    ckpt_path = os.path.join(save_dir, f"flowmatch_ckpt_epoch_{epoch:03d}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"[saved checkpoint] {ckpt_path}")

print("Training finished.")