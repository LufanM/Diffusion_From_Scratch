import math 
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# conda env is ddpm_study

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, pred, target, weighted = 1.0):
        loss = self._loss(pred, target)
        WeightedLoss = loss * weighted
        return WeightedLoss

class WightedL1(WeightedLoss):
    def _loss(self, pred, target):
        return torch.abs(pred - target)

class WightedL2(WeightedLoss):
    def _loss(self, pred, target):
        """
        reduction='mean'等价于torch.nn.MSELoss()默认的reduction方式，即对所有元素求平均值，返回一个标量；
        如果设置为'reduction='sum''，则对所有元素求和；
        如果设置为'reduction='none''，则不进行任何缩减，返回与输入相同形状的张量，其中每个元素都是对应位置的损失值。
        """
        return F.mse_loss(pred, target, reduction='mean')   
    
Losses = {
    'l1': WightedL1,
    'l2': WightedL2,
}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    # print(f"out.shape: {out.shape}") # t 维度为bathch_size, out.shape也是(batch_size,)
    # print(f"out.reshape: {out.reshape(b, *((1,) * (len(x_shape) - 1)))}")
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).to(device)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):  # 可以替换成更复杂的网络结构，比如Transformer或者UNets
    def __init__(self,state_dim, action_dim, device, t_dim=16):
        super(MLP, self).__init__()

        self.device = device
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        self.output_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=-1)
        x = self.mid_layer(x)

        return self.output_layer(x)


class Diffusion(nn.Module):
    def __init__(self, loss_type='l2', beta_schedule="linear", clip_denoised=True,  predict_epsilon=True, **kwargs):
        super().__init__()
        self.state_dim = kwargs['obs_dim']
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.T = kwargs['T']
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.device = torch.device(kwargs['device'])
        self.model = MLP(self.state_dim, self.action_dim, self.device, self.T)

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # [1,2,3]-> [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        #前向过程
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        #反向过程 
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        # 为什么要使用log？因为在采样过程中需要计算方差的平方根，如果方差过小，可能会导致数值不稳定，使用log可以避免这种情况，同时也方便计算。
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20))) # log(0) = -inf, clamp to avoid this

        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))   
        self.register_buffer('sqrt_recipm_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)) # calculate x0 from xt
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )

        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance


    def predict_start_from_noise(self, x, t, pred_noise): # calculate x0 from xt and noise
        return (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise)
        #第一项张量是[256,1],第二个张量是[256,2]，广播机制自动扩展第一项的维度，使其与第二项的维度匹配，从而进行逐元素的计算。

    def p_mean_variance(self, x, t, s):
        pred_noise = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon.clamp_(-1, 1) # 这里作简化，实际应以环境的action space为准进行裁剪
        # calculate the mean and variance of p(x_{t-1} | x_t) using the predicted x0 and the current xt，笔记中作了代还，消掉了x0，直接用xt和噪声计算了均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t) 
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, s)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,)*(len(x.shape)-1)))  # no noise when t == 0)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, *args, **kwargs):
        device = self.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state) # 每一步采样，输入当前的xt和时间步t，输出下一个xt-1，直到t=0，得到最终的采样结果x0

        return x


    def sample(self, state, *args, **kwargs):
        """
        state: [batch_size, state_dim]
        """
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp(-1, 1)


    # ------------------------------------------training------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise) # 前向扩散，根据当前的x0和时间步t，生成对应的xt
        x_recon = self.model(x_noisy, t, state) # 输出模型预测出的噪声（MLP的前向传播）

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)


        return loss

    def loss(self, x, state, weight=1.0):
        batch_size = len(x)
        # 均匀分布随机采样一个时间步t，作为训练的目标时间步，这样模型就能学习在不同时间步下的噪声预测能力，从而提高模型的泛化能力。与论文一致
        t = torch.randint(0, self.T, (batch_size,), device=x.device).long()  
        return self.p_losses(x, state, t, weight)


    # -------------------------------------------inference------------------------------------------
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(256, 2).to(device)
    state = torch.randn(256, 11).to(device)
    model = Diffusion(loss_type='l2', obs_dim=11, action_dim=2, hidden_dim=256, T=100, device=device).to(device)
    action = model(state)

    loss = model.loss(x, state)

    print(f"action: {action}, loss: {loss}")