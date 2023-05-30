import random
import numpy as np
from PIL import Image, ImageFilter



def salt_and_pepper_noise(image, probability, intensity):
    width, height = image.size
    pixels = image.load()

    # 计算要添加噪声的像素数量
    num_noise_pixels = int(probability * width * height)

    # 在随机位置添加椒盐噪声
    for _ in range(num_noise_pixels):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # 随机选择像素值为黑色或白色
        if random.random() < intensity:
            pixels[x, y] = (0, 0, 0)  # 黑色
        else:
            pixels[x, y] = (255, 255, 255)  # 白色

    return image


def gaussian_noise(image, mean, std_dev):
    width, height = image.size
    pixels = np.array(image)

    # 生成高斯噪声
    noise = np.random.normal(mean, std_dev, (height, width, 3))
    noisy_image = np.clip(pixels + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_image)


def blur_noise(image, radius):
    return image.filter(ImageFilter.BoxBlur(radius))

