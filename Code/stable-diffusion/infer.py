import torch
import cv2
from PIL import Image
import PIL

import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

prompt = "woman's dress high quality"
image = pipe(prompt).images[0]

image.save("/home/adeeshb/Adeesh/VLR/project/VLR_Project_Stable_Diffusion/Code/stable-diffusion/img1.jpg")