# Code to test proof of concept of mask augmentation for stable diffusion 

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the results folder if it doesn't exist

def AugmentMask():
    if not os.path.exists('results'):
        os.makedirs('results')
        
    
    input_images_path = '/home/adeeshb/Adeesh/VLR/project/VLR_Project_Stable_Diffusion/Code/cloth-segmentation/input_images/fashion_sample1.jpg'
    output_images_path = '/home/adeeshb/Adeesh/VLR/project/VLR_Project_Stable_Diffusion/Code/cloth-segmentation/output_images/fashion_sample1.jpg'
    
    img1_path = '/home/adeeshb/Adeesh/VLR/project/VLR_Project_Stable_Diffusion/Code/cloth-segmentation/input_images/fashion_sample1.jpg'
    img2_path = '/home/adeeshb/Adeesh/VLR/project/VLR_Project_Stable_Diffusion/Code/cloth-segmentation/output_images/fashion_sample1.jpg'
    
    # for input_image in os.listdir(input_images_path):
    # if (input_image.endswith('.jpg') or input_image.endswith('.png')):
    #     img1_path  = input_images_path + input_image
    #     img2_path = output_images_path + input_image
        

        # # Read the input images
        # img1 = cv2.imread('.\inputs\sample1_inputs\sample1.jpg')
        # img2 = cv2.imread('.\inputs\sample1_inputs\sample1_segmentation.jpg')

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Resize img2 to match img1 size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert the images to float32 for calculations
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Loop over all masks in the texture folder
    texture_folder = '/home/adeeshb/Adeesh/VLR/project/VLR_Project_Stable_Diffusion/Code/inputs/cloth_textures'
    for mask_file in os.listdir(texture_folder):
        if mask_file.endswith('.jpg') or mask_file.endswith('.png'):
            # Read the mask image
            mask = cv2.imread(os.path.join(texture_folder, mask_file))
            mask = cv2.resize(mask, (img1.shape[1], img1.shape[0]))
            mask = mask.astype(np.float32)
            
            # Choose pixel values from img1 where mask is non-zero, else from img2
            result = np.where(img2,mask,img1)

            # Save the augmented image
            # import pdb; pdb.set_trace();
            result_file = os.path.splitext(mask_file)[0] + '_result.jpg'
            cv2.imwrite(os.path.join('results', result_file), result)

            # Display and save the images
            fig, axs = plt.subplots(1, 4, figsize=(10,10))

            axs[0].imshow(cv2.cvtColor(img1/255.0, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Input Image')
            axs[0].set_axis_off()

            axs[1].imshow(cv2.cvtColor(img2/255.0, cv2.COLOR_BGR2RGB))
            axs[1].set_title('Segmentation Mask')
            axs[1].set_axis_off()

            axs[2].imshow(cv2.cvtColor(mask/255.0, cv2.COLOR_BGR2RGB))
            axs[2].set_title('Cloth Texture')
            axs[2].set_axis_off()

            axs[3].imshow(cv2.cvtColor(result/255.0, cv2.COLOR_BGR2RGB))
            axs[3].set_title('Augmented Image')
            axs[3].set_axis_off()

            plt.savefig(os.path.join('results', os.path.splitext(mask_file)[0] + '_pipeline.jpg'))
            plt.show()

if __name__ == '__main__':
    AugmentMask()

