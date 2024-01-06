#this code to generate a 8x8 img to present
from PIL import Image
import os
import argparse
def merge_images(input_path, output_path):
    images = []
    subfolders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]

    input_folder=os.path.join(input_path,subfolders[0])
 
    for i in range(1,65):  
        image_path = os.path.join(input_folder, f"{i}.png") 
        image = Image.open(image_path)
        images.append(image)

 
    width, height = images[0].size
    total_width = width * 8
    total_height = height * 8

 
    new_image = Image.new("RGB", (total_width, total_height))


    for i in range(8):
        for j in range(8):
            index = i * 8 + j
            new_image.paste(images[index], (j * width, i * height))


    new_image.save(output_path)
    print(f"Merged image saved to: {output_path}")

parser = argparse.ArgumentParser()
parser.add_argument("--model",default='face')
args = parser.parse_args()

input_folder = '/home/cs3964_group2/yuxiaoyang/final_proj/fake'
input_folder=os.path.join(input_folder,args.model)
output_path = f'./{args.model}.png'
merge_images(input_folder, output_path)
