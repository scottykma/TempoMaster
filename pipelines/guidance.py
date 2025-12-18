import torch
import torch.nn.functional as F


class FlowMatchingGuidance:
    def __init__(self, 
                 guidance_scale: float = 8,
                 use_apg: bool = True,
                 eta: float = 0.0,
                 norm_threshold: float = 27.0,
                 momentum: float = -0.5
        ):

        self.guidance_scale = guidance_scale
        self.use_apg = use_apg
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.momentum = momentum
        
        self.momentum_buffer = None
        if momentum != 0:
            self.momentum_buffer = MomentumBuffer(momentum)
    
    def __call__(self, noise_cond, noise_uncond, z_t, t):
        D_cond = z_t - t * noise_cond
        D_uncond = z_t - t * noise_uncond
        
        if self.use_apg:
            guided_D = self._apply_apg(D_cond, D_uncond)
        else:
            guided_D = D_uncond + self.guidance_scale * (D_cond - D_uncond)
        
        epsilon = 1e-8
        guided_noise = (z_t - guided_D) / (t + epsilon)
        
        return guided_noise
    
    def _apply_apg(self, D_cond, D_uncond):
        delta_D = D_cond - D_uncond

        if self.momentum_buffer is not None:
            self.momentum_buffer.update(delta_D)
            delta_D = self.momentum_buffer.running_average
    
        if self.norm_threshold > 0:
            delta_D = self._rescale(delta_D)
        
        delta_D_parallel, delta_D_orthogonal = self._project(delta_D, D_cond)
        
        delta_D_adjusted = delta_D_orthogonal + self.eta * delta_D_parallel
        
        guided_D = D_cond + (self.guidance_scale - 1) * delta_D_adjusted
        
        return guided_D
    
    def _project(self, v0, v1):
        original_dtype = v0.dtype

        v0 = v0.double()
        v1 = v1.double()
        
        v1_normalized = F.normalize(v1.flatten(1), dim=1).view_as(v1)
        
        # [B, C, T, H, W]
        batch_size = v0.shape[0]
        v0_flat = v0.flatten(1)  # [B, C*T*H*W]
        v1_flat = v1_normalized.flatten(1)  # [B, C*T*H*W]
        
        dot_product = (v0_flat * v1_flat).sum(dim=1, keepdim=True)  # [B, 1]
        
        v0_parallel_flat = dot_product * v1_flat
        v0_parallel = v0_parallel_flat.view_as(v0)
        
        v0_orthogonal = v0 - v0_parallel

        return v0_parallel.to(original_dtype), v0_orthogonal.to(original_dtype)
    
    def _rescale(self, delta_D):
        norm = delta_D.norm(p=2, dim=(1, 3, 4), keepdim=True) # C, H, W
       
        scale_factor = torch.minimum(
            torch.ones_like(norm),
            self.norm_threshold / (norm + 1e-8)
        )
        
        return delta_D * scale_factor
    


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = None
    
    def update(self, update_value: torch.Tensor):
        if self.running_average is None:
            self.running_average = update_value.clone()
        else:
            self.running_average = update_value + self.momentum * self.running_average
    
    def reset(self):
        self.running_average = None

