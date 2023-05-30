import argparse
import sys
import os
import yaml

from pathlib import Path
from PIL import Image
from noise import salt_and_pepper_noise, gaussian_noise, blur_noise


noise_list = ['salt_and_pepper', 'gaussian', 'blur']

# 获取当前文件夹路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# 命令行参数
def parse_opt(known=False):
    # 创建解析器对象
    parser = argparse.ArgumentParser()

    # 可选参数
    parser.add_argument('--input', type=str, default=ROOT / 'input', help='input path to the images directory')
    parser.add_argument('--output', type=str, default=ROOT / 'output', help='output path to the images directory')
    parser.add_argument('--noise', type=str, choices=noise_list, default='salt_and_pepper', help='noise type')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def add_noise(image_path, output_path, noise_type, noise_data):
    image = Image.open(image_path)
    filename = os.path.basename(image_path)

    if noise_type == 'salt_and_pepper':
        image = salt_and_pepper_noise(image, noise_data['salt_and_pepper']['probability'],
                                      noise_data['salt_and_pepper']['intensity'])
    elif noise_type == 'gaussian':
        image = gaussian_noise(image, noise_data['gaussian']['mean'], noise_data['gaussian']['std_dev'])
    elif noise_type == 'blur':
        image = blur_noise(image, noise_data['blur']['radius'])
    else:
        raise ValueError(f'Invalid noise type: {noise_type}')

    output_file = os.path.join(output_path, filename)
    image.save(output_file)


def process_images(input_folder, output_folder, noise_type, noise_data):
    # 检查保存路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('using', noise_type, 'noise')
    
    num_images = 0  # 图片计数变量

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            add_noise(image_path, output_folder, noise_type, noise_data)
            num_images += 1 # 每处理一张图片，计数加一

    print(f"A total of {num_images} images were processed")


def main(opt):
    with open(os.path.join(ROOT, 'config.yaml'), 'r') as file:
        noise_data = yaml.safe_load(file)

    process_images(opt.input, opt.output, opt.noise, noise_data)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
