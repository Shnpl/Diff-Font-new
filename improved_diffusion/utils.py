import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def resize_image(img, resolution):
    while min(*img.size) >= 2 * resolution:
        img = img.resize(
            tuple(x // 2 for x in img.size), resample=Image.Resampling.BOX
        )

    scale = resolution / min(*img.size)
    img = img.resize(
        tuple(round(x * scale) for x in img.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(img.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    arr = arr[crop_y: crop_y + resolution, crop_x: crop_x + resolution]
    arr = arr.astype(np.float32) / 127.5 - 1

    transf = transforms.ToTensor()
    img = transf(arr)
    return img