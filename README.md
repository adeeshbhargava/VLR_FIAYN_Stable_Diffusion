# VLR_Project_Stable_Diffusion
This project is aims towards Generating Pose Aware Photo-Realistic Clothes For A Target Image, using an input image of the cloth texture, and a text prompt.

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

# Libraries Needed:
cv2 ,matplotlib, numpy , os 


# Step0: Make sure image is 512x512 resoltuion and in .jpg format

# Step1: Cloth Segmentation:
Path: Code/cloth-segmentation
1) Create an input_images folder
2) Crete an output_images folder
3) Download pretrained model from this [link](https://drive.google.com/file/d/1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ/view?usp=sharing)(165 MB) in `trained_checkpoint` folder.
4) Put input images in `input_images` folder
5) Run `python infer.py` for inference.
6) Output will be saved in `output_images`

# Step2: Mask Augmentation Code:
Path: Code/mask-augmentation.py
1) Add input images folders path
2) Add segmentation Mask path
3) Run `python mask_augmentation.py` and results would be saved in 'results' folder

# Step3 : Run Stable Diffusion 2.1
Path: Code/Stable-Diffusion-Custom/utils/pipeline.py
1) check for the 'image_path_dir' and 'mask_path' inside 'Inpaint' function
2) initialise the right stable diffusion model - (`mad3310/stable-diffusion-fashion-v1-1,stabilityai/stable-diffusion-2-1,CompVis/stable-diffusion-v1-4,CompVis/stable-diffusion-v1-4`) are a few viable options and pretrained models
3) Results will be saved in `Code/diffusion_results` folder

# Step4: Run PIDM on results of Stable Diffusion
Path: Code/PIDM

## Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n PIDM python=3.6
conda activate PIDM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

The folder structure of any custom dataset should be as follows:

- dataset/
- - <dataset_name>/
- - - img/ 
- - - pose/
- - - train_pairs.txt
- - - test_pairs.txt

You basically will have all your images inside ```img``` folder. You can use different subfolders to store your images or put all your images inside the ```img``` folder as well. The corresponding poses are stored inside ```pose``` folder (as txt file if you use openpose. In our project, we use 18-point keypoint estimation). ```train_pairs.txt``` and ```test_pairs.txt``` will have paths of all possible pairs seperated by comma ```<src_path1>,<tgt_path1>```.

After that, run the following command to process the data:

```
python data/prepare_data.py \
--root ./dataset/<dataset_name> \
--out ./dataset/<dataset_name>
--sizes ((256,256),)
```

This will create an lmdb dataset ```./dataset/<dataset_name>/256-256/```

## Inference 

Download the pretrained model from [here](https://drive.google.com/file/d/1WkV5Pn-_fBdiZlvVHHx_S97YESBkx4lD/view?usp=share_link) and place it in the ```checkpoints``` folder.
For pose control use ```obj.predict_pose``` as in the following code snippets. 

  ```python
from predict import Predictor
obj = Predictor()

obj.predict_pose(image=<PATH_OF_SOURCE_IMAGE>, sample_algorithm='ddim', num_poses=4, nsteps=50)

  ```

For apperance control use ```obj.predict_appearance```

  ```python
from predict import Predictor
obj = Predictor()

src = <PATH_OF_SOURCE_IMAGE>
ref_img = <PATH_OF_REF_IMAGE>
ref_mask = <PATH_OF_REF_MASK>
ref_pose = <PATH_OF_REF_POSE>

obj.predict_appearance(image=src, ref_img = ref_img, ref_mask = ref_mask, ref_pose = ref_pose, sample_algorithm = 'ddim',  nsteps = 50)

  ```

The output will be saved as ```output.png``` filename.

