
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# Create the results folder if it doesn't exist
if not os.path.exists('diffusion_results_compiled'):
    os.makedirs('diffusion_results_compiled')

# Get the list of all files in the results directory
results_dir = './results/'
results_files = os.listdir(results_dir)

#Read Input image
input_image = cv2.imread('D:\CMU ACADS\Spring23\Visual Learning and Recognition\project\Code\inputs\sample1_inputs\sample1.jpg')

# Loop over all the files in the results directory
for file_name in results_files:
    # Check if the file is a texture result
    if 'result' in file_name:
        # Get the texture number
        #import pdb; pdb.set_trace();
        texture_file_name = file_name.split('_')[0]+'_'+file_name.split('_')[1] 
        
        # Read the texture result image
        texture_result_path = os.path.join(results_dir, file_name)
        texture_result = cv2.imread(texture_result_path)
        
        # Get the corresponding diffusion result file name
        diffusion_result_name = texture_file_name + '_diffusion_result.jpg'
        
        # Read the diffusion result image
        diffusion_result_path = os.path.join('./diffusion results/', diffusion_result_name)
        diffusion_result = cv2.imread(diffusion_result_path)
        
        #Read the texture image
        texture_name= texture_file_name + '.jpg'
        texture_path = os.path.join('./inputs/cloth_textures/', texture_name)
        texture_prompt = cv2.imread(texture_path)

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))

        # Display the first image in the first subplot
        axs[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Input Image')
        axs[0].set_axis_off()
        
        axs[1].imshow(cv2.cvtColor(texture_prompt, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Style Prompt')
        axs[1].set_axis_off()
        
        axs[2].imshow(cv2.cvtColor(texture_result, cv2.COLOR_BGR2RGB))
        axs[2].set_title('Augmented Image')
        axs[2].set_axis_off()

        # Display the second image in the second subplot
        axs[3].imshow(cv2.cvtColor(diffusion_result, cv2.COLOR_BGR2RGB))
        axs[3].set_title('Diffusion Result')
        axs[3].set_axis_off()

        # Set the title of the figure
        fig.suptitle('Fashion Is All You Need: Stable Diffusion Results')

        # Show the figure
        plt.savefig(os.path.join('diffusion_results_compiled', texture_file_name  + '_diffusion_combined.jpg'))
        plt.show()