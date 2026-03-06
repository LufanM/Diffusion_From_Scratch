import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(self, model, T=1000, device='cuda'):
        super().__init__()
        self.model  = model # 也就是 UNet 模型，功能与diffusion_sample中的MLP一样
        self.T      = T
        self.device = device

        betas               = torch.linspace(0.0001, 0.02, T, dtype=torch.float32)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance  = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',           alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',           torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance',            posterior_variance)
        self.register_buffer('sqrt_recip_alphas_cumprod',     torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm_alphas_cumprod',    torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # -------- 前向加噪 --------
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod,           t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # -------- 训练损失 --------
    def loss(self, x_start):
        B       = x_start.shape[0]
        t       = torch.randint(0, self.T, (B,), device=x_start.device).long()
        noise   = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        pred_noise = self.model(x_noisy, t)

        return F.mse_loss(pred_noise, noise)  # 标量 ✅

    # -------- 单步去噪 --------
    @torch.no_grad()
    def p_sample(self, x, t):
        B        = x.shape[0]
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

        pred_noise = self.model(x, t_tensor) # 一般要引入状态s，笔记中是MLP(x,t,s)，这里是UNet(x,t)，因为图像生成不需要状态输入

        x_recon = (
            extract(self.sqrt_recip_alphas_cumprod,  t_tensor, x.shape) * x -
            extract(self.sqrt_recipm_alphas_cumprod, t_tensor, x.shape) * pred_noise
        )
        x_recon = x_recon.clamp(-1, 1)

        posterior_mean = (
            extract(self.posterior_mean_coef1, t_tensor, x.shape) * x_recon +
            extract(self.posterior_mean_coef2, t_tensor, x.shape) * x
        )

        if t == 0:
            return posterior_mean

        noise    = torch.randn_like(x)
        variance = extract(self.posterior_variance, t_tensor, x.shape).sqrt()
        return posterior_mean + variance * noise

    # -------- 完整采样 --------
    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, t)
        return x