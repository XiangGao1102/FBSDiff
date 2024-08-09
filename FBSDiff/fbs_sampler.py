import torch
import numpy as np
from tqdm import tqdm
from FBSDiff.dct_util import dct_2d, idct_2d, low_pass, high_pass
from FBSDiff.ddim_sampler import DDIM_Sampler


class FBS_Sampler(DDIM_Sampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super(FBS_Sampler, self).__init__(model, schedule, **kwargs)

    @torch.no_grad()
    def decode_with_low_FBS(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                                 unconditional_conditioning, use_original_steps=False, callback=None,
                                 threshold=90, end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(ref_latent)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            if step > end_step:
                x_dec_dct = dct_2d(x_dec, norm='ortho')
                ref_latent_dct = dct_2d(ref_latent, norm='ortho')
                merged_dct = low_pass(ref_latent_dct, threshold) + high_pass(x_dec_dct, threshold+1)
                x_dec = idct_2d(merged_dct, norm='ortho')

                ref_latent, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=1.0,
                                                 unconditional_conditioning=None)
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning)
            else:
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def decode_with_high_FBS(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                                 unconditional_conditioning, use_original_steps=False, callback=None,
                                 threshold=20, end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(ref_latent)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            if step > end_step:
                x_dec_dct = dct_2d(x_dec, norm='ortho')
                ref_latent_dct = dct_2d(ref_latent, norm='ortho')
                merged_dct = high_pass(ref_latent_dct, threshold) + low_pass(x_dec_dct, threshold-1)
                x_dec = idct_2d(merged_dct, norm='ortho')

                ref_latent, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=1.0,
                                                 unconditional_conditioning=None)
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning)
            else:
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec

    @torch.no_grad()
    def decode_with_mid_FBS(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                                 unconditional_conditioning, use_original_steps=False, callback=None,
                                 threshold1=20, threshold2=40, end_step=500):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = torch.randn_like(ref_latent)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            if step > end_step:
                x_dec_dct = dct_2d(x_dec, norm='ortho')
                ref_latent_dct = dct_2d(ref_latent, norm='ortho')
                merged_dct = low_pass(x_dec_dct, threshold1) + high_pass(low_pass(ref_latent_dct, threshold2), threshold1+1) + high_pass(x_dec_dct, threshold2+1)
                x_dec = idct_2d(merged_dct, norm='ortho')

                ref_latent, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=1.0,
                                                 unconditional_conditioning=None)
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                              unconditional_guidance_scale=unconditional_guidance_scale,
                                              unconditional_conditioning=unconditional_conditioning)
            else:
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
