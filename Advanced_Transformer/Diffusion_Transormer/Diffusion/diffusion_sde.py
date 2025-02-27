import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, t_dim=16):
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class SDEBase(abc.ABC):
    def __init__(self, T):
        self.T = T
        self.dt = 1 / T

    @abc.abstractmethod
    def drift(self, x_t, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x_t, t):
        pass

    def dw(self, x):
        return torch.randn_like(x) * math.sqrt(self.dt)

    def reverse_ode(self, x, t, score):
        dx = (self.drift(x, t) - 0.5 * self.dispersion(x, t) ** 2 * score) * self.dt
        return x - dx

    def reverse_sde(self, x, t, score):
        dx = (
            self.drift(x, t) - self.dispersion(x, t) ** 2 * score
        ) * self.dt + self.dispersion(x, t) * self.dw(x) * (t > 0)
        return x - dx

    def forward_step(self, x, t):
        dx = self.drift(x, t) * self.dt + self.dispersion(x, t) * self.dw(x)
        return x + dx

    def forward(self, x_0):
        x_ = x_0
        for t in range(self.T):
            x_ = self.forward_step(x_, t)
        return x_

    def reverse(self, x_t, score, mode, state):
        for t in reversed(range(self.T)):
            score_value = score(x_t, torch.full((x_t.shape[0],), t, dtype=torch.long), state)
            if mode == "sde":
                x_t = self.reverse_sde(x_t, t, score_value)
            elif mode == "ode":
                x_t = self.reverse_ode(x_t, t, score_value)
        return x_t


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class MySDE(SDEBase):
    def __init__(self, T, schedule):
        super().__init__(T=T)

        if schedule == "vp":
            self.thetas = vp_beta_schedule(T)

        self.sigmas = torch.sqrt(2 * self.thetas)

        thetas_cumsum = torch.cumsum(self.thetas, dim=0)
        # self.dt = -math.log(0.1) / thetas_cumsum[-1]

        self.thetas_bar = thetas_cumsum * self.dt
        self.vars = 1 - torch.exp(-2 * self.thetas_bar)
        self.stds = torch.sqrt(self.vars)

    def drift(self, x_t, t):
        return -self.thetas[t] * x_t

    def dispersion(self, x_t, t):
        return self.sigmas[t]

    def compute_score_from_noise(self, noise, t):
        return -noise / self.stds[t]

    def generate_random_state(self, a_0):
        noise = torch.randn_like(a_0)
        t = torch.randint(0, self.T, (a_0.shape[0], 1)).long()
        a_t = a_0 * torch.exp(-self.thetas_bar[t]) + self.stds[t] * noise
        return a_t, t

    def ground_truth_score(self, a_t, t, a_0):
        return (a_0 * torch.exp(-self.thetas_bar[t]) - a_t) / self.vars[t]


class DiffusionSDEPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, T, max_action):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.max_action = max_action

        self.model = MLP(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.sde = MySDE(T, 'vp')

    def score_fn(self, a_t, t, state):
        noise = self.model(a_t, t, state)
        return self.sde.compute_score_from_noise(noise, t[0])

    def sample(self, state, mode):
        noise = torch.randn(state.shape[0], self.action_dim)
        action = self.sde.reverse(noise, self.score_fn, mode=mode, state=state)
        return action.clamp_(-self.max_action, self.max_action)

    def loss(self, state, a_0):
        a_t, t = self.sde.generate_random_state(a_0)
        score_pred = self.sde.compute_score_from_noise(
            self.model(a_t, t.squeeze(1), state), t
        )
        score_true = self.sde.ground_truth_score(a_t, t, a_0)
        return F.mse_loss(score_pred, score_true)


def test_diffusion_sde_policy():
    # 初始化超参数
    state_dim = 8
    action_dim = 4
    hidden_dim = 128
    T = 10
    max_action = 1.0

    # 创建 DiffusionSDEPolicy 实例
    policy = DiffusionSDEPolicy(state_dim, action_dim, hidden_dim, T, max_action)

    # 假设的输入数据
    batch_size = 16
    state = torch.randn(batch_size, state_dim)  # 随机生成状态
    action = torch.randn(batch_size, action_dim)  # 随机生成动作

    # 测试前向传播
    print("Testing forward pass (sampling)...")
    sampled_action = policy.sample(state, mode="sde")
    print("Sampled action:", sampled_action)

    # 测试损失计算
    print("Testing loss computation...")
    loss = policy.loss(state, action)
    print("Loss:", loss.item())

# 运行测试
if __name__ == "__main__":
    test_diffusion_sde_policy()
