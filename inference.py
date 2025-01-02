import einops
import numpy as np
import random
import torch
from PIL import Image
import os
from pytorch_lightning import seed_everything
from FBSDiff.tools import create_model, load_state_dict
from FBSDiff.fbs_sampler import FBS_Sampler

# resolution of the generated image
H = W = 512

# number of different images generated for a given reference image and a given text prompt
num_samples = 1

# random seed
seed = -1

# set the total steps of the inversion trajectory
encode_steps = 1000

# set the total steps of the sampling trajectory
decode_steps = 100

# set the value of lambda (0~1), the larger the value, the shorter the calibration phase.
lambda_end = 0.5

# the end step of the calibration phase
end_step = encode_steps * lambda_end

ddim_eta = 0

unconditional_guidance_scale = 7.5

# set the image path of the reference image
img_path = 'test.jpg'

# set the target text prompt
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


# generate translated image with low-FBS
    th_lp = 90  # set the low-pass filtering threshold th_lp
    x_rec = sampler.decode_with_low_FBS(ref_latent=latent, cond=cond, t_dec=decode_steps,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=un_cond, threshold=th_lp,
                                                 end_step=end_step)

# generate translated image with high-FBS
#     th_hp = 3   # set the high-pass filtering threshold th_hp
#     x_rec = sampler.decode_with_high_FBS(ref_latent=latent, cond=cond, t_dec=decode_steps,
#                                                         unconditional_guidance_scale=unconditional_guidance_scale,
#                                                         unconditional_conditioning=un_cond, threshold=th_hp,
#                                                         end_step=end_step)


# generate translated image with mid-FBS
#     th_mp1 = 5   # set the mid-pass filtering lower bound threshold
#     th_mp2 = 80  # set the mid-pass filtering upper bound threshold
#     x_rec = sampler.decode_with_mid_FBS(ref_latent=latent, cond=cond, t_dec=decode_steps,
#                                                        unconditional_guidance_scale=unconditional_guidance_scale,
#                                                        unconditional_conditioning=un_cond, threshold1=th_mp1,
#                                                        threshold2=th_mp2,
#                                                        end_step=end_step)

    x_samples = torch.clip(model.decode_first_stage(x_rec), min=-1, max=1)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(
        np.uint8)

    for sample in x_samples:
        Image.fromarray(sample).save('res.jpg')
