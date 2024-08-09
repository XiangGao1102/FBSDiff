import einops
import numpy as np
import random
import torch
from PIL import Image
import os
from pytorch_lightning import seed_everything
from FBSDiff.tools import create_model, load_state_dict
from FBSDiff.fbs_sampler import FBS_Sampler

H = W = 512
num_samples = 1
seed = -1
encode_steps = 1000
decode_steps = 100
lambda_end = 0.5
end_step = encode_steps * lambda_end
ddim_eta = 0
unconditional_guidance_scale = 7.5
img_path = 'test.jpg'

target_prompt = 'painting of ancient ruins'

model = create_model('./models/model_ldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('./models/v1-5-pruned-emaonly.ckpt', location='cuda'), strict=False)
sampler = FBS_Sampler(model)

img = np.array(Image.open(img_path).resize((H, W)))
img = (img.astype(np.float32) / 127.5) - 1.0          # -1 ~ 1
img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].repeat(num_samples, 1, 1, 1).cuda()  # n, 3, 512, 512

with torch.no_grad():
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    un_cond = {"c_crossattn": [model.get_learned_conditioning([''] * num_samples)]}
    cond = {"c_crossattn": [model.get_learned_conditioning([target_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if not os.path.exists('latent.pt'):
        encoder_posterior = model.encode_first_stage(img_tensor)
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        sampler.make_schedule(ddim_num_steps=encode_steps)
        latent, out = sampler.encode(x0=z, cond=un_cond, t_enc=encode_steps)
        torch.save(latent, 'latent.pt')

    sampler.make_schedule(ddim_num_steps=decode_steps)

    latent = torch.load('latent.pt').cuda()

    x_rec = sampler.decode_with_low_FBS(ref_latent=latent, cond=cond, t_dec=decode_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=un_cond, threshold=90,
                                                 end_step=end_step)

    # x_rec = sampler.decode_with_high_FBS(ref_latent=latent, cond=cond, t_dec=decode_steps,
    #                                                     unconditional_guidance_scale=unconditional_guidance_scale,
    #                                                     unconditional_conditioning=un_cond, threshold=3,
    #                                                     end_step=end_step)

    # x_rec = sampler.decode_with_mid_FBS(ref_latent=latent, cond=cond, t_dec=decode_steps,
    #                                                    unconditional_guidance_scale=unconditional_guidance_scale,
    #                                                    unconditional_conditioning=un_cond, threshold1=5,
    #                                                    threshold2=80,
    #                                                    end_step=end_step)

    x_samples = torch.clip(model.decode_first_stage(x_rec), min=-1, max=1)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(
        np.uint8)

    for sample in x_samples:
        Image.fromarray(sample).save('res.jpg')