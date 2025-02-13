import torch
import torch.nn as nn
import enum
import torch.nn.functional as F
import numpy as np
import math

###
### x_t = (alpha_hat_t) * x_0 + (1-alpha_hat_t) * random_noise

#class ScheduleSelect(enum):
#    LinearScheduler: 1
#    CosineSchedular: 2


class BetaScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def forward(self):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.timesteps)

class LinearScheduler(BetaScheduler):
    def __init__(self, timesteps):
        super().__init__(timesteps)

    def forward(self):
        scale = 1000 / self.timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float64)
        return betas
    
class CosineScheduler(BetaScheduler):
    def __init__(self, timesteps, s=0.008):
        super().__init__(timesteps)
        self.s = s
    
    def forward(self):
        steps = self.timesteps + 1
        x = np.linspace(0, self.timesteps, steps)
        alpha_cumprod = np.cos(((x / self.timesteps) + self.s) / (1 + self.s) * np.pi * 0.5) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
        
class GaussianDiffusion:
    def __init__(self, timesteps, beta_schedular = 'linear'):
        
        self.betas = LinearScheduler(timesteps).forward()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), 'constant', value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_inverse_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_inverse_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_inversem1_alphas_cumprod = torch.sqrt(1.0 / (self.alphas_cumprod - 1.0))
        
        self.posterier_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterier_variance = torch.clamp(self.posterier_variance, min=1e-20)
        self.log_posterience_variance = torch.log(self.posterier_variance)
        self.posterier_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterier_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    @staticmethod
    def _extract(vals, t, x_shape):
        batch_size = t.shape[0]
        output = vals.gather(-1, t.cpu())
        return output.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_diffusion_sample(self, original_sample, t, device="cpu"):
        noise = torch.randn_like(original_sample, device=original_sample.device, dtype=original_sample.dtype)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, original_sample.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, original_sample.shape)
        
        return sqrt_alphas_cumprod_t * original_sample + sqrt_one_minus_alphas_cumprod_t * noise

    def backward_diffusion_sample(self, latents, model_output, t):
        sqrt_inverse_alphas_cumprod_t = self._extract(self.sqrt_inverse_alphas_cumprod, t, latents.shape)
        sqrt_inversem1_alphas_cumprod_t = self._extract(self.sqrt_inversem1_alphas_cumprod, t, latents.shape)
        
        pred_original_sample = sqrt_inverse_alphas_cumprod_t * model_output + sqrt_inversem1_alphas_cumprod_t * latents
        pred_previous_sample = self._extract(self.posterier_mean_coef1, t, latents.shape) * pred_original_sample + self._extract(self.posterier_mean_coef2, t, latents.shape) * latents
        
        log_posterior_variance_t = self._extract(self.log_posterience_variance, t, latents.shape)
        
        variance = 0
        if t > 0:
            noise = torch.rand_like(model_output)
            variance = torch.exp(0.5 * log_posterior_variance_t) * noise
            
        return pred_previous_sample + variance
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        embeds = time[:, None] * torch.exp(torch.arange(self.dim // 2, device = device) * -(math.log(10000.0) / (self.dim // 2 - 1)))[None, :]
        embeds = torch.cat([torch.sin(embeds), torch.cos(embeds)], dim=-1)
        return embeds
    
class AttentionBlock(nn.Module):
    def __init__(self, units, groups=8):
        super().__init__()
        self.units = units
        self.groups = groups
        
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=units)
        self.q = nn.Linear(units, units)
        self.k = nn.Linear(units, units)
        self.v = nn.Linear(units, units)
        self.out = nn.Linear(units, units)
        
    def forward(self, x):
        batch_size, height, width, channels = x.shape
        scale = self.units ** -0.5
        
        x = self.norm(x)
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        attn_score = torch.einsum('bhwc,bijc->bhwij', q, k) * scale 
        attn_score = torch.reshape(attn_score, (batch_size, height, width, height * width))
        
        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = torch.reshape(attn_score, (batch_size, height, width, height, width))
        
        proj = torch.einsum('bhwij,bijc->bhwc', attn_score, v)
        proj = self.out(proj)
        return x + proj
    
class ResidualBlock(nn.Module):
    def __init__(self, units, groups=8):
        super().__init__()
        self.units = units
        self.groups = groups
        
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=units)
        self.conv1 = nn.Conv2d(units, units, 1)
        self.linear1 = nn.Linear(units, units)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=units)
        self.conv2 = nn.Conv2d(units, units, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(units, units, kernel_size=3, padding=1)
        
    def forward(self, inputs):
        x, t = inputs
        input_width = x.shape[3]
        if input_width == self.units:
            residual = x
        else:
            residual = self.conv1(x)
            
        temb = F.silu(t)
        temb = self.linear1(inputs)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = x + temb
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual
    
class DownSample(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.conv = nn.Conv2d(units, units, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    def __init__(self, width, interpolation="nearest"):
        super().__init__()
        self.width = width
        self.interpolation = interpolation
        
        # Define the upsample layer using PyTorch's interpolate function in the forward method
        self.conv = nn.Conv2d(
            in_channels=width, out_channels=width, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        if self.interpolation == "nearest":
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        elif self.interpolation == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation}")
        
        x = self.conv(x)
        return x
    
class TimeMLP(nn.Module):
    def __init__(self, units, num_layers):
        super().__init__()
        self.units = units
        self.num_layers = num_layers

        self.mlp = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, units)
        )
        
    def forward(self, x):
        for _ in range(self.num_layers):
            x = self.mlp(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, 
                 img_size, 
                 img_channels, 
                 widths, 
                 has_attention, 
                 num_res_blocks=2,
                 norm_groups=8,
                 interpolation="nearest"):
        super(UNet, self).__init__()
        self.conv_channels = widths[0] 
        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.conv_channels,
            kernel_size=3,
            padding=1
        )
        
        self.temb = PositionalEncoding(dim=self.conv_channels*4)
        self.temp_mlp = TimeMLP(units=self.conv_channels*4)
        
        self.skips = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        
        # DownBlock
        for i in range(len(widths)):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(widths[i], groups=norm_groups))
                if has_attention[i]:
                    self.down_blocks.append(AttentionBlock(widths[i], groups=norm_groups))
            if widths[i] != widths[-1]:
                self.down_blocks.append(DownSample(widths[i]))
        
        # MiddleBlock
        self.middle_block1 = ResidualBlock(widths[-1], groups=norm_groups)
        self.middle_attention = AttentionBlock(widths[-1], groups=norm_groups)
        self.middle_block2 = ResidualBlock(widths[-1], groups=norm_groups)
        
        # UpBlock
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(widths))):
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(nn.Conv2d(widths[i] * 2, widths[i], kernel_size=3, padding=1))
                self.up_blocks.append(ResidualBlock(widths[i], groups=norm_groups))
                if has_attention[i]:
                    self.up_blocks.append(AttentionBlock(widths[i], groups=norm_groups))
            if i != 0:
                self.up_blocks.append(UpSample(widths[i], interpolation=interpolation))
        
        self.final_norm = nn.GroupNorm(num_groups=norm_groups, num_channels=widths[0])
        self.final_conv = nn.Conv2d(widths[0], 3, kernel_size=3, padding=1)
        
    def forward(self, image_input, time_input):
        x = self.initial_conv(image_input)
        temb = self.temb(time_input)
        temb = self.temb_mlp(temb)
        self.skips.append(x)
        
        # DownBlock
        for down_block in self.down_blocks:
            x = down_block([x, temb])
            self.skips.append(x)
        
        # MiddleBlock
        x = self.middle_block1([x, temb])
        x = self.middle_attention(x)
        x = self.middle_block2([x, temb])
        
        # UpBlock
        for up_block in self.up_blocks:
            x = torch.cat([x, self.skips.pop()], dim=1)
            x = up_block([x, temb])
        
        x = self.final_norm(x)
        x = self.activation_fn(x)
        x = self.final_conv(x)
        
        return

        
        
if __name__ == "__main__":
    gss = GaussianDiffusion(1000)


