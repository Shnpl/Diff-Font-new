import sys
sys.path.append('..')
from PIL import Image,ImageFont,ImageDraw
import torchvision.transforms as transforms
import torch
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
from random import Random
import os
def load_data(
    *, data_dir, batch_size, image_size, deterministic=False,style_dir=None,char_set=None
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    dataset = ImageDataset(
        resolution=image_size,
        data_dir = data_dir,
        style_dir=style_dir,
        char_set=char_set
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

class ImageDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 resolution=64, 
                 char_set=None,
                 stroke_path:str = 'datasets/CFG/new_strokes.json',
                 style_dir:str = None):
        super().__init__()
        if not data_dir:
            raise ValueError("unspecified data directory")
        if char_set == None:
            pass
        elif type(char_set) == str:
            with open(char_set,'r') as f:
                char_set = json.load(f)
        elif type(char_set) == list:
            pass
        else:
            raise TypeError("char_set should be path or list")
        all_files = []
        secondary_dirs = os.listdir(data_dir)
        secondary_dirs = [os.path.join(data_dir,style_dir) for style_dir in secondary_dirs]
        
        for secondary_dir in secondary_dirs:
            files = os.listdir(secondary_dir)
            if char_set:
                files = [file for file in files if os.path.splitext(file)[0] in char_set]

            files = [os.path.join(secondary_dir,file) for file in files if os.path.splitext(file)[-1] in [".jpg", ".jpeg", ".png", ".gif"]]
            all_files.extend(files) 

        self.resolution = resolution
        self.local_images = all_files
        #self.local_classes = None if classes is None else classes[shard:][::num_shards]
        # self.local_styles = None if styles is None else styles[shard:][::num_shards]
        self.use_stroke = False
        self.content_ttf_path = "datasets/CFG/font_content.ttf"
        self.transform_img = resizeKeepRatio((128,128))
        self.style_dir = style_dir
        if stroke_path != None:
            self.use_stroke = True
            with open(stroke_path,'r') as f:
                self.strokes = json.load(f)
            self.stroke_part_list = [
            "㇔",# 点 \u31d4
            "㇐",# 横 \u31d0
            "㇑",# 竖 \u31d1
            "㇓",# 撇 \u31d3
            "㇏",# 捺 \u31cf
            "㇀",# 提 \u31c0
            "𡿨",# 撇点 \ud847\udfe8
            "㇙",# 竖提 \u31d9
            "㇊",# 横折提 \u31ca
            "㇁",# 弯钩 \u31c1
            "㇚",# 竖钩 \u31da
            "㇟",# 竖弯钩 \u31df
            "㇂",# 斜钩 \u31c2
            "㇃",# 卧钩 \u31c3
            "㇖",# 横沟 \u31d6
            "㇆",# 横折钩 \u31c6
            "㇈",# 横折弯钩 \u31c8
            "㇌",# 横撇弯钩 \u31cc
            "㇡",# 横折折折钩 \u31e1
            "㇉",# 竖折折钩 \u31c9
            "㇗",# 竖弯 \u31d7
            "㇍",# 横折弯 \u31cd
            "㇕",# 横折 \u31d5
            "𠃊",# 竖折 \ud840\udcca
            "ㄥ",# 撇折 \u3125
            "㇇",# 横撇 \u31c7
            "㇋",# 横折折撇 \u31cb
            "ㄣ",# 竖折撇 \u3123
            "⺄",# 横斜钩 \u2e84
            "㇞",# 竖折折 \u31de
            "㇅",# 横折折 \u31c5
            "㇎" # 横折折折 \u31ce
            ]
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        img_basename = os.path.basename(path)
        img_name = os.path.splitext(img_basename)[0]
        context_dict = {}
        with bf.BlobFile(path, "rb") as f:
            img_PIL = Image.open(f)
            img_PIL.load()
        if self.use_stroke:
            char_strokes = self.strokes[img_name]
            stroke_count_dict = dict([(stroke_part,0) for stroke_part in self.stroke_part_list])
            for stroke in char_strokes:
                stroke_count_dict[stroke] += 1
            stroke_list = []
            for stroke in self.stroke_part_list:
                stroke_list.append(stroke_count_dict[stroke])
            stroke_arr =  np.array(stroke_list,dtype=np.float32)
        
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        scale = self.resolution / min(*img_PIL.size)
        img_PIL = img_PIL.resize(
            tuple(round(x * scale) for x in img_PIL.size), resample=Image.BICUBIC
        )

        arr = np.array(img_PIL.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        if self.style_dir:
            current_style = os.path.dirname(path).split('/')[-1]
            style_encoding_path = os.path.join(self.style_dir,f"{current_style}.pt")
            style_img_encoding = torch.load(style_encoding_path,map_location='cpu')
            style_img_encoding = torch.unsqueeze(style_img_encoding,dim=0)
            context_dict["style_image"] = style_img_encoding
        else:
            # Get another Image from the same style for the stlye encoder
            current_style_path = os.path.dirname(path)
            current_style_imgs = os.listdir(current_style_path)
            start = 0
            end = len(current_style_imgs)-1
            current_style_img = current_style_imgs[Random().randint(start,end)]

            style_path = os.path.join(current_style_path,current_style_img)

            with bf.BlobFile(style_path, "rb") as f:
                style_img_128_PIL = Image.open(f)
                style_img_128_PIL.load()
                style_image_128_arr = np.array(style_img_128_PIL.convert("RGB"))
            context_dict['style_name'] = current_style_path.split('/')[-1]
            context_dict["style_image"] = np.transpose(style_image_128_arr, [2, 0, 1])
            # We are not on a new enough PIL to support the `reducing_gap`
            # argument, which uses BOX downsampling at powers of two first.
            # Thus, we do it by hand to improve downsample quality.
            while min(*style_img_128_PIL.size) >= 2 * 128:
                style_img_128_PIL = style_img_128_PIL.resize(
                    tuple(x // 2 for x in style_img_128_PIL.size), resample=Image.BOX
                )

            scale = 128 / min(*style_img_128_PIL.size)
            style_img_128_PIL = style_img_128_PIL.resize(
                tuple(round(x * scale) for x in style_img_128_PIL.size), resample=Image.BICUBIC
            )

            style_image_128_arr = np.array(style_img_128_PIL.convert("RGB"))
            crop_y128 = (style_image_128_arr.shape[0] - 128) // 2
            crop_x128 = (style_image_128_arr.shape[1] - 128) // 2
            style_image_128_arr = style_image_128_arr[crop_y128 : crop_y128 + 128, crop_x128 : crop_x128 + 128]
            style_image_128_arr = style_image_128_arr.astype(np.float32) / 127.5 - 1

        ##
        # content
        font = ImageFont.truetype(self.content_ttf_path, 80)
        
        try:
            content_image_128_PIL = Image.new('RGB', (128, 128), (255, 255, 255))
            drawBrush = ImageDraw.Draw(content_image_128_PIL)
            drawBrush.text((0, 0), img_name, fill=(0, 0, 0), font=font)
            content_image_128_arr = self.transform_img(content_image_128_PIL)

        except:
            raise
        ##

        
        content_unicode = ord(img_name)
        context_dict["content_text"] = np.array(content_unicode, dtype=np.int64)
        context_dict["content_image"] = content_image_128_arr
        
        if self.use_stroke:
            context_dict["stroke"] = stroke_arr
        return np.transpose(arr, [2, 0, 1]), context_dict


class resizeKeepRatio(object):

    def __init__(self, size, interpolation=Image.BILINEAR, 
        train=False):

        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.train = train

    def __call__(self, img):

        if img.mode == 'L':
            img_result = Image.new("L", self.size, (255))
        elif img.mode =='RGB':
            img_result = Image.new("RGB",self.size, (255, 255, 255))
        else:
            print("Unknow image mode!")

        img_w, img_h = img.size

        target_h = self.size[1]
        target_w = max(1, int(img_w * target_h / img_h))

        if target_w > self.size[0]:
            target_w = self.size[0]

        img = img.resize((target_w, target_h), self.interpolation)
        #begin = random.randint(0, self.size[0]-target_w) if self.train else int((self.size[0]-target_w)/2)
        begin = int((self.size[0]-target_w)/2)

        box = (begin, 0, begin+target_w, target_h)
        img_result.paste(img, box)

        img = self.toTensor(img_result)
        img.sub_(0.5).div_(0.5)
        return img