from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,use_stroke = False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    with open('datasets/CFG/seen_characters.json', 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    all_files = _list_image_files_recursively(data_dir, json_data)
    classes = None
    # styles = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        # styles = [bf.dirname(path) for path in all_files]
    # for index in [5209]:
    #     for key, value in sorted_classes.items():
    #         if value == index:
    #             print(key)
    if use_stroke:
        stroke_path = "datasets/CFG/new_strokes.json"
    else:
        stroke_path = None
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        # styles=styles,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        stroke_path=stroke_path
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


def _list_image_files_recursively(data_dir, json_data):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"] and entry.split(".")[0] in json_data:
        # if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path, json_data))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, styles=None, shard=0, num_shards=1,stroke_path:str = None):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        # self.local_styles = None if styles is None else styles[shard:][::num_shards]
        self.use_stroke = False
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
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        if self.use_stroke:
            char_strokes = self.strokes[path.split("/")[-1].split(".")[0]]
            stroke_count_dict = dict([(stroke_part,0) for stroke_part in self.stroke_part_list])
            for stroke in char_strokes:
                stroke_count_dict[stroke] += 1
            style_stroke_list = []
            for stroke in self.stroke_part_list:
                style_stroke_list.append(stroke_count_dict[stroke])
            style_stroke_arr =  np.array(style_stroke_list,dtype=np.float32)
        
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        with bf.BlobFile(path, "rb") as f:
            pil_image128 = Image.open(f)
            pil_image128.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image128.size) >= 2 * 128:
            pil_image128 = pil_image128.resize(
                tuple(x // 2 for x in pil_image128.size), resample=Image.BOX
            )

        scale = 128 / min(*pil_image128.size)
        pil_image128 = pil_image128.resize(
            tuple(round(x * scale) for x in pil_image128.size), resample=Image.BICUBIC
        )

        arr128 = np.array(pil_image128.convert("RGB"))
        crop_y128 = (arr128.shape[0] - 128) // 2
        crop_x128 = (arr128.shape[1] - 128) // 2
        arr128 = arr128[crop_y128 : crop_y128 + 128, crop_x128 : crop_x128 + 128]
        arr128 = arr128.astype(np.float32) / 127.5 - 1

        out_dict = {}
        out_sty = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # if self.local_styles is not None:
        #     out_sty["style"] = self.local_styles[idx]
        # return np.transpose(arr, [2, 0, 1]), out_dict, out_sty
        if self.use_stroke:
            return np.transpose(arr, [2, 0, 1]), out_dict, np.transpose(arr128, [2, 0, 1]),style_stroke_arr
        else:
            return np.transpose(arr, [2, 0, 1]), out_dict, np.transpose(arr128, [2, 0, 1])
