# VLR_Project_Stable_Diffusion
This project is aimed towards generating new clothes for target image using style and text prompts from the user.

# Abstract:
The fashion industry stands to benefit greatly from the use of AI-generated fashion trends, which can provide personalized fashion recommendations based on an individual's body type, skin tone, and personal preferences, ultimately transforming the way people perceive and approach fashion. Our work focuses on generating pose-aware, photo-realistic clothes for a target image, which takes into account the visual context to ensure accuracy and aesthetic appeal. We use a combination of cloth segmentation, augmentation, and a stable diffusion model that considers the context of the prompt to generate high-quality results. We also utilize PIDM (Person Image Synthesis via Denoising Diffusion Model)\cite{bhunia2023person} to generate images of the person wearing the clothing item in different poses to ensure proper alignment with the body in the target image. Our approach produces realistic-looking clothing items that accurately reflect the texture of the input image while maintaining the integrity of the target image's pose and bodily features.

# Results:

# Pipeline for Augmentation + Stable Diffusion:
![result](https://user-images.githubusercontent.com/116693172/236724536-e049c6b9-0693-4efb-92f0-1057608b0d91.jpg)

# Adding Custom Poses + Target Dress Texture:
<img width="1352" alt="Pipeline" src="https://user-images.githubusercontent.com/116693172/236724772-89bcdffe-748a-402a-aa1b-700358a96c5e.png">

# Results For various Target Textures:
![texture_0_diffusion_combined](https://user-images.githubusercontent.com/116693172/236724392-1270c567-4574-4fc3-8ba0-4a13514dcad2.jpg)
![texture_1_diffusion_combined](https://user-images.githubusercontent.com/116693172/236724395-bb37c876-c59f-405c-834f-042cddf2123a.jpg)
![texture_2_diffusion_combined](https://user-images.githubusercontent.com/116693172/236724396-ce9db99c-91bc-40b0-9e4c-813dab434375.jpg)
![texture_3_diffusion_combined](https://user-images.githubusercontent.com/116693172/236724397-7d3725bc-01b8-485e-8518-df7fab1a9572.jpg)
![texture_4_diffusion_combined](https://user-images.githubusercontent.com/116693172/236724398-29fa9717-c15a-4f9b-b50f-0a404e158910.jpg)


# Setup Required:

Libraries Needed:
cv2 ,matplotlib, numpy , os 

# Mask Augmentation Code:
