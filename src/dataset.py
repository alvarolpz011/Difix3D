import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F


def _load_and_normalize_depth(path, image_size):
    depth = np.array(Image.open(path)).astype(np.float32)
    depth = depth / depth.max() if depth.max() > 0 else depth
    depth_t = torch.from_numpy(depth).unsqueeze(0)
    depth_t = F.resize(depth_t, image_size)
    depth_t = (depth_t - 0.5) / 0.5
    return depth_t


def _prepare_rgb(path, image_size):
    img = Image.open(path).convert("RGB")
    img_t = F.to_tensor(img)
    img_t = F.resize(img_t, image_size)
    img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
    return img_t


def _build_dataset_from_dir(root_dir, split, default_prompt="remove degradation"):
    scene_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    data = {}
    random.seed(42)
    for scene_dir in sorted(scene_dirs):
        test_dir = os.path.join(scene_dir, "test")
        if not os.path.isdir(test_dir):
            continue
        rgb_paths = sorted([p for p in os.listdir(test_dir) if re.match(r"r_\\d+\\.png", p)])
        if not rgb_paths:
            continue
        num_train = int(len(rgb_paths) * 0.9)
        selected = rgb_paths[:num_train] if split == "train" else rgb_paths[num_train:]
        for idx, rgb_name in enumerate(selected):
            rgb_path = os.path.join(test_dir, rgb_name)
            depth_name = rgb_name.replace(".png", "_depth_0001.png")
            depth_path = os.path.join(test_dir, depth_name) if os.path.exists(os.path.join(test_dir, depth_name)) else None

            ref_name = selected[(idx + 1) % len(selected)] if len(selected) > 1 else rgb_name
            ref_path = os.path.join(test_dir, ref_name)
            ref_depth_name = ref_name.replace(".png", "_depth_0001.png")
            ref_depth_path = os.path.join(test_dir, ref_depth_name) if os.path.exists(os.path.join(test_dir, ref_depth_name)) else None

            data_id = f"{os.path.basename(scene_dir)}_{rgb_name.replace('.png', '')}"
            data[data_id] = {
                "image": rgb_path,
                "target_image": rgb_path,
                "ref_image": ref_path,
                "prompt": default_prompt,
            }
            if depth_path is not None:
                data[data_id]["depth_image"] = depth_path
            if ref_depth_path is not None:
                data[data_id]["ref_depth_image"] = ref_depth_path
    return data


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None, default_prompt="remove degradation", use_depth_conditioning=False):
        super().__init__()

        if os.path.isfile(dataset_path):
            with open(dataset_path, "r") as f:
                self.data = json.load(f)[split]
        else:
            self.data = _build_dataset_from_dir(dataset_path, split, default_prompt=default_prompt)

        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.use_depth_conditioning = use_depth_conditioning

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        item = self.data[img_id]

        input_img = item["image"]
        output_img = item["target_image"]
        ref_img = item.get("ref_image")
        caption = item.get("prompt", "remove degradation")
        depth_img = item.get("depth_image") if self.use_depth_conditioning else None
        ref_depth_img = item.get("ref_depth_image") if self.use_depth_conditioning else None

        img_t = _prepare_rgb(input_img, self.image_size)
        output_t = _prepare_rgb(output_img, self.image_size)

        if self.use_depth_conditioning:
            if depth_img is not None:
                depth_t = _load_and_normalize_depth(depth_img, self.image_size)
            else:
                depth_t = torch.zeros(1, *self.image_size)
            img_t = torch.cat([img_t, depth_t], dim=0)

        cond_views, target_views = [img_t], [output_t]

        if ref_img is not None:
            ref_t = _prepare_rgb(ref_img, self.image_size)
            if self.use_depth_conditioning:
                if ref_depth_img is not None:
                    ref_depth_t = _load_and_normalize_depth(ref_depth_img, self.image_size)
                else:
                    ref_depth_t = torch.zeros(1, *self.image_size)
                ref_t = torch.cat([ref_t, ref_depth_t], dim=0)

            cond_views.append(ref_t)
            target_views.append(_prepare_rgb(ref_img, self.image_size))

        if len(cond_views) == 1:
            cond_views.append(cond_views[0])
            target_views.append(target_views[0])

        conditioning = torch.stack(cond_views, dim=0)
        output = torch.stack(target_views, dim=0)

        out = {
            "output_pixel_values": output,
            "conditioning_pixel_values": conditioning,
            "caption": caption,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
