import numpy as np
from PIL import Image

# 读取.npz文件
file_path = 'samples_10x256x256x3.npz'
loaded_data = np.load(file_path)

# 获取名为'samples'的数组
samples = loaded_data['arr_0']

# 遍历每张图片
for i in range(samples.shape[0]):
    # 从数组中获取一张图片
    image_data = samples[i]

    # 创建PIL Image对象
    image = Image.fromarray(image_data.astype(np.uint8))

    # 保存图片
    image.save(f'image_{i + 1}.png')

# 关闭文件
loaded_data.close()

# >>> arr0 = data['arr_0']
# >>> arr1 = data['arr_1']
# >>> arr0.shape
# (10, 256, 256, 3)
# >>> arr1.shape
# (10,)
