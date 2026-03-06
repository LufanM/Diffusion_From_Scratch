import torch
import torch.nn as nn
import math


# ========== 时间嵌入 ==========
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # [B, dim]


# ========== ResNet 块 ==========
class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.s0 = nn.Sequential(
            nn.GroupNorm(8, dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, 3, padding=1)
        )

        self.s1 = nn.Sequential(
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )

        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        t = self.time_mlp(time_emb)[:, :, None, None]  # [B, dim_out, 1, 1]
        h = self.s0(x) + t
        h = self.s1(h)
        return h + self.res(x)


# ========== Self-Attention 块 ==========
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm  = nn.GroupNorm(8, dim)
        self.qkv   = nn.Conv2d(dim, dim * 3, 1)
        self.proj  = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, -1).transpose(1, 2)  # [B, HW, C]
        k = k.reshape(B, C, -1)                  # [B, C, HW]
        v = v.reshape(B, C, -1).transpose(1, 2)  # [B, HW, C]

        attn = torch.bmm(q, k) * self.scale       # [B, HW, HW]
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v)                  # [B, HW, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


# ========== 下采样 / 上采样 ==========
class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):  # ✅ 修正：支持通道数变化
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):  # ✅ 修正：支持通道数变化
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


# ========== U-Net ==========
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_dim=64,
        dim_mults=(1, 2, 4, 8),
        time_dim=256,
    ):
        super().__init__()

        # ---------- 时间嵌入 ----------
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # dims = [64, 128, 256, 512]
        dims   = [base_dim * m for m in dim_mults]
        # in_out = [(64,128), (128,256), (256,512)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ---------- 输入层 ----------
        # 3 → 64
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # ---------- Encoder ----------
        # 每层结构：
        #   ResnetBlock(dim_in → dim_in)   保存 skip1
        #   ResnetBlock(dim_in → dim_in)   保存 skip2
        #   AttentionBlock
        #   Downsample(dim_in → dim_out)   ✅ 在下采样时扩展通道
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = (i == len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_dim),
                ResnetBlock(dim_in, dim_in, time_dim),
                AttentionBlock(dim_in),
                # ✅ 修正：下采样时同时扩展通道数
                # 最后一层不下采样，用普通卷积扩通道
                Downsample(dim_in, dim_out) if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # ---------- Bottleneck ----------
        # 输入：dims[-1] = 512
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn   = AttentionBlock(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        # ---------- Decoder ----------
        # 每层结构：
        #   Concat(当前 + skip2) → ResnetBlock(dim_out+dim_in → dim_out)
        #   Concat(当前 + skip1) → ResnetBlock(dim_out+dim_in → dim_out)
        #   AttentionBlock
        #   Upsample(dim_out → dim_in)  ✅ 上采样时恢复通道数
        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (i == len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                # ✅ 修正：Concat 后的通道数 = dim_out + dim_in（skip 是 dim_in）
                ResnetBlock(dim_out + dim_in, dim_out, time_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_dim),
                AttentionBlock(dim_out),
                # ✅ 修正：上采样时恢复通道数
                Upsample(dim_out, dim_in) if not is_last
                else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        # ---------- 输出层 ----------
        # Decoder 最后输出 dims[0] = 64 通道
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, dims[0]),  # ✅ 修正：用 dims[0] 而不是 base_dim（其实一样，但更明确）
            nn.SiLU(),
            nn.Conv2d(dims[0], in_channels, 1)
        )

    def forward(self, x, t):
        # x: [B, 3, H, W]
        # t: [B]

        # 时间嵌入
        t_emb = self.time_mlp(t)   # [B, time_dim]

        # 输入层
        x = self.conv_in(x)        # [B, 64, H, W]

        # ---------- Encoder ----------
        # 每层保存 2 个 skip（res1 和 res2 的输出）
        skips = []
        for res1, res2, attn, down in self.downs:
            x = res1(x, t_emb)
            skips.append(x)        # 保存 skip1 [B, dim_in, H, W]
            x = res2(x, t_emb)
            x = attn(x)
            skips.append(x)        # 保存 skip2 [B, dim_in, H, W]
            x = down(x)            # [B, dim_out, H/2, W/2]

        # ---------- Bottleneck ----------
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # ---------- Decoder ----------
        for res1, res2, attn, up in self.ups:
            # ✅ pop 顺序和 Encoder push 顺序相反
            x = torch.cat([x, skips.pop()], dim=1)  # Concat skip2
            x = res1(x, t_emb)
            x = torch.cat([x, skips.pop()], dim=1)  # Concat skip1
            x = res2(x, t_emb)
            x = attn(x)
            x = up(x)              # 上采样并恢复通道数

        return self.conv_out(x)    # [B, in_channels, H, W]


# ========== 测试 ==========
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(
        in_channels=3,
        base_dim=64,
        dim_mults=(1, 2, 4, 8),
        time_dim=256,
    ).to(device)

    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    out = model(x, t)
    print(f"输入: {x.shape}")   # [2, 3, 32, 32]
    print(f"输出: {out.shape}") # [2, 3, 32, 32] ✅