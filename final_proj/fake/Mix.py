#park
import os
import shutil

def copy_images(source_dir, destination_dir, num_images):
    # Get the list of image files in the source directory
    image_files = [file for file in os.listdir(source_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Sort the image files alphabetically
    image_files.sort()

    # Copy and rename the first num_images files to the destination directory
    for i, image_file in enumerate(image_files[:num_images]):
        # print(source_dir, i)
        if i < 100:
            continue
        source_path = os.path.join(source_dir, image_file)
        filename, extension = os.path.splitext(image_file)
        # destination_filename = f"{os.path.basename(source_dir)}_{i+1}{extension}"
        destination_filename = f"mix_{i+1}{extension}"
        destination_path = os.path.join(destination_dir, destination_filename)
        shutil.copy2(source_path, destination_path)

def rename_files(directory):
    # Get the list of files in the directory
    files = os.listdir(directory)

    # Rename the files as {i}.png
    for i, file in enumerate(files):
        file_path = os.path.join(directory, file)
        new_file_name = f"mix_{i+1}.png"
        # print(i)
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(file_path, new_file_path)

# Specify the source directories
dir_1 = "/home/cs3964_group2/data/proj_dataset"
dir_2 = "/home/cs3964_group2/yuxiaoyang/final_proj/fake/o100/samples_o100"

# Specify the destination directory
# dir_3 = "/home/cs3964_group2/yuxiaoyang/final_proj/fake/GAN+o100/GAN+o100_dataset"
dir_3 = "/home/cs3964_group2/yuxiaoyang/final_proj/fake/GAN/GAN_dataset"

# Specify the number of images to copy from each directory
n = 500  # Number of images from dir_1
m = 100   # Number of images from dir_2

# Create the destination directory if it doesn't exist
os.makedirs(dir_3, exist_ok=True)

# Copy the images from dir_1
copy_images(dir_1, dir_3, n)

# Copy the images from dir_2
# copy_images(dir_2, dir_3, m)

# rename_files(dir_3)