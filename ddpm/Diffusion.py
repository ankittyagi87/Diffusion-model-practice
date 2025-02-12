import torch
import torchvision
import enum
import torch.nn.functional as F

###
### x_t = (alpha_hat_t) * x_0 + (1-alpha_hat_t) * random_noise

#class ScheduleSelect(enum):
#    LinearScheduler: 1
#    CosineSchedular: 2


class BetaScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps

    def forward(self):
        pass

class LinearScheduler(BetaScheduler):
    def __init__(self, timesteps):
        super().__init__(timesteps)

    def forward(self):
        scale = 1000 / self.timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float64)
        return betas

class GaussianDiffusion:
    def __init__(self, timesteps, beta_schedular = 'linear'):
        
        self.betas = LinearScheduler(timesteps).forward()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), 'constant', value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        




if __name__ == "__main__":
    gss = GaussianDiffusion(1000)


