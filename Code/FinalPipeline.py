import os
import sys
import cv2
import subprocess

def RunClothesSegmentation():
    # Running segmentation on the given image
    path_segmentation_model = "cloth-segmentation/infer.py"
    subprocess.call(["python", path_segmentation_model])
    return True

def RunMaskAugmentation():
    # Running segmentation on the given image
    #path_segmentation_model = "cloth-segmentation/infer.py"
    subprocess.call(["python", "mask_augmentation.py"])


if __name__ == "__main__":
    if(RunClothesSegmentation()):
        RunMaskAugmentation()