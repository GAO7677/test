#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import random
import atexit
import getpass
import shutil
import tarfile
import time
import sys

import argparse
import copy
import gc
import json
import logging
import math
import os
import pickle
import random
import shutil
import string
import warnings
from os.path import join as ospj
from pathlib import Path
from torchvision.transforms import functional as Func

import diffusers
import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
)
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from util import natural_keys, CLASSNAMES, TEMPLATES_SMALL

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__)


def int2none(v):
    if v is None or v == "":
        return None
    elif v.lower() in ("none", "null"):
        return None
    else:
        return int(v)


def str2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ("none", "null"):
        return None
    else:
        return str(v)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    elif v is None or v == "":
        return None
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_local():
    yaml_file = os.environ.get("CHIMERA_CONFIG", "local.yaml")
    with open(yaml_file, "r") as f:
        args_local = {k: os.path.expandvars(str(v)) for k, v in yaml.safe_load(f).items()}
    return args_local


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--n_shot", type=int2none, default=16)
    parser.add_argument("--n_template", type=int2none, default=1)
    parser.add_argument("--fewshot_seed", type=str, default="seed0", help="seed{number}.")
    parser.add_argument("--target_class_idx", type=int2none, default=None, help="if None, it is datset-wise method.")
    parser.add_argument("--instance_idx", type=int2none, default=None, help="if not None, it is LoFT method.")
    parser.add_argument("--is_tqdm", type=str2bool, default=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16")
    parser.add_argument("--fewshot_dir", type=str, default=None, help="A folder containing the training data of instance images.")
    parser.add_argument("--validation_epochs", type=int, default=1000, help="Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--center_crop", type=str2bool, default=False, help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    parser.add_argument("--train_unet", type=str2bool, default=True, help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--train_text_encoder", type=str2bool, default=True, help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int2none, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=10000, help="Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `\"latest\"` to automatically select the last available checkpoint.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate_A", type=float, default=5e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--learning_rate_B", type=float, default=5e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="The scheduler type to use. Choose between [\"linear\", \"cosine\", \"cosine_with_restarts\", \"polynomial\", \"constant\", \"constant_with_warmup\"].")
    parser.add_argument("--lr_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="""[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.""")
    parser.add_argument("--allow_tf32", action="store_true", help="""Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices""")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="""The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.""")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="""Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.""")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--tokenizer_max_length", type=int, default=None, required=False, help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.")
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true", required=False, help="Whether to use attention mask for the text encoder")
    parser.add_argument("--rank", type=int, default=16, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--step", type=int, default=50, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--method", type=str, default="JOINT", help="The dimension of the LoRA update matrices.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # set local arguments
    args_local = set_local()
    args.output_dir = args_local["output_dir"]
    args.fewshot_dir = args_local["fewshot_dir"]
    args.mask_dir = args_local["mask_dir"] 
    if args.pretrained_model_name_or_path is None:
        args.pretrained_model_name_or_path = args_local["model_dir"]

    # set output directory
    mid0 = "chimera_rank" + str(args.rank) + "_ep" + str(args.num_train_epochs) + "_" + str(args.method)
    mid1 = f"shot{args.n_shot}_{args.fewshot_seed}"
    mid2 = (
        "" if args.target_class_idx is None else 
        CLASSNAMES[args.dataset][args.target_class_idx]
    )
    args.output_dir = ospj(
        args.output_dir,
        args.dataset,
        mid0,
        mid1,
        mid2,
    )

    mid_dir = f"shot{args.n_shot}_{args.fewshot_seed}"

    if "novel" in args.dataset:
        args.fewshot_dir = ospj(
            args.fewshot_dir, args.dataset.removesuffix("_novel"), mid_dir
        )
        args.mask_dir = ospj(args.mask_dir, args.dataset.removesuffix("_novel"), f"real/shot{args.n_shot}_{args.fewshot_seed}")
    else:
        args.fewshot_dir = ospj(
            args.fewshot_dir, args.dataset, mid_dir
        )
        args.mask_dir = ospj(args.mask_dir, args.dataset, f"real/shot{args.n_shot}_{args.fewshot_seed}")

    args.instance_fname = None
    args.instance_fname_ext = None
    if args.instance_idx is not None:
        if args.target_class_idx is None:
            raise ValueError("`target_class_idx` should be defined when `instance_idx` is defined.")

        classname = CLASSNAMES[args.dataset][args.target_class_idx]
        data_dir = ospj(args.fewshot_dir, classname)
        filename = ".".join(os.listdir(data_dir)[args.instance_idx].split(".")[:-1])
        file_ext = os.listdir(data_dir)[args.instance_idx].split(".")[-1]
        args.output_dir = ospj(args.output_dir, filename)
        args.instance_fname = filename
        args.instance_fname_ext = file_ext

        if os.path.exists(ospj(args.output_dir, "pytorch_lora_weights.safetensors")):
            print("skip ", args.output_dir)
            sys.exit(0)

    else:
        if os.path.exists(ospj(args.output_dir, "B0_each", "pytorch_lora_weights.safetensors")):
            print("skip ", args.output_dir)
            sys.exit(0)
        
    return args


def clean_up_shm(target_dir):
    """ clean up when a single job is remained when Python exits """

    # check job count
    count_job_fpath = ospj(target_dir, "count_job.json")
    with open(count_job_fpath, "r") as f:
        count_job = json.load(f)
    
    if count_job == 1:
        shutil.rmtree(target_dir, ignore_errors=True)
        print(f"Cleaned up data path {target_dir}.")

    else:
        count_job -= 1
        with open(count_job_fpath, "w") as f:
            json.dump(count_job, f)
        print(f"Did not clean up data path, {count_job} jobs are still running!")


def extract_tarfile(classname, source_dir, target_dir=None, delete_tarfile=False):
    tar_fpath = ospj(source_dir, f"{classname}.tar")

    # directory where the file exist
    extract_dir = source_dir if target_dir is None else target_dir
    if "/" in classname:
        extract_dir = ospj(extract_dir, *classname.split("/")[:-1])
        os.makedirs(extract_dir, exist_ok=True)
    
    # extract tar file
    with tarfile.open(tar_fpath, 'r') as tar:
        tar.extractall(extract_dir)
        if delete_tarfile:
            os.remove(tar_fpath)


def sanitize_box(box, W, H):
    """Clamp and order the box to [x1,y1,x2,y2] within image bounds."""
    x1, y1, x2, y2 = box
    x1 = float(min(max(0, x1), W)); x2 = float(min(max(0, x2), W))
    y1 = float(min(max(0, y1), H)); y2 = float(min(max(0, y2), H))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def per_axis_requirements(W, H, cx, cy, S):
    half = S / 2.0
    dx = min(cx, W - cx)
    dy = min(cy, H - cy)
    s_req_x = float("inf") if dx <= 0 else (half / dx)
    s_req_y = float("inf") if dy <= 0 else (half / dy)
    return s_req_x, s_req_y


def s_max_for_bbox(bw, bh, S):
    return S / max(1e-12, max(bw, bh))


def choose_scale_one_side_pad(W, H, box, S, rng, eps=1e-6):
    x1, y1, x2, y2 = sanitize_box(box, W, H)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        raise ValueError("Invalid bbox.")

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s_req_x, s_req_y = per_axis_requirements(W, H, cx, cy, S)
    s_max_bbox = s_max_for_bbox(bw, bh, S)

    a = min(s_req_x, s_req_y)
    b = max(s_req_x, s_req_y)

    # Try exact one-side band
    lo_one_side = a
    hi_one_side = min(s_max_bbox, b - eps)

    if math.isfinite(lo_one_side) and lo_one_side <= hi_one_side:
        s_min_used = lo_one_side
        s_max_used = hi_one_side
        s = rng.uniform(s_min_used, s_max_used)
        return s, True, s_min_used, s_max_used

    # Fallback band (exact one-side impossible)
    s_min_used = min(s_max_bbox, max((a if math.isfinite(a) else 0.0) * 0.8, 0.5 * s_max_bbox))
    s_max_used = s_max_bbox
    if s_max_used < s_min_used:
        s_min_used = s_max_used  # degenerate
    s = rng.uniform(s_min_used, s_max_used) if s_max_used > s_min_used else s_max_used
    return s, False, s_min_used, s_max_used


def box_has_area(box, W, H):
    x1, y1, x2, y2 = sanitize_box(box, W, H)
    return (x2 - x1) > 0 and (y2 - y1) > 0


def crop_centered(image, box, S, s, pad_color=(0,0,0)):
    """Scale image by s, then extract an SxS crop centered at the scaled bbox center. Pad if needed."""
    W, H = image.width, image.height
    new_w = max(1, int(round(W * s)))
    new_h = max(1, int(round(H * s)))
    scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    x1, y1, x2, y2 = sanitize_box(box, W, H)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scx, scy = cx * s, cy * s

    half = S / 2.0
    pad_left   = max(0, int(math.ceil(half - scx)))
    pad_top    = max(0, int(math.ceil(half - scy)))
    pad_right  = max(0, int(math.ceil(scx + half - new_w)))
    pad_bottom = max(0, int(math.ceil(scy + half - new_h)))

    if any(v > 0 for v in (pad_left, pad_top, pad_right, pad_bottom)):
        canvas = Image.new(
            "RGB",
            (new_w + pad_left + pad_right, new_h + pad_top + pad_bottom),
            pad_color,
        )
        canvas.paste(scaled, (pad_left, pad_top))
        scaled = canvas
        scx += pad_left
        scy += pad_top

    left = int(round(scx - half))
    top  = int(round(scy - half))
    return scaled.crop((left, top, left + S, top + S))


def augment_image_by_spec(image: Image.Image, box: list, target_size: int = 512):
    """
    Augments an image by dynamically scaling, padding if necessary, and then always cropping.
    This version is robust and handles all geometric cases elegantly.
    """
    # --- 1. Initial Resize (Common for all cases) ---
    w, h = image.width, image.height
    if w < h:
        base_w = target_size
        base_h = int(target_size * h / w)
    else:
        base_h = target_size
        base_w = int(target_size * w / h)
    
    base_image = image.resize((base_w, base_h), Image.Resampling.BILINEAR)

    # --- 2. Logic Branch based on box validity ---
    if box == [0, 0, 0, 0]:
        # Perform a simple random crop on the base image.
        top = random.randint(0, base_h - target_size)
        left = random.randint(0, base_w - target_size)
        final_image = Func.crop(base_image, top, left, target_size, target_size)

    else:
        # --- 3. Calculate dynamic scale for the object ---
        resize_factor = base_w / w
        base_box = [int(coord * resize_factor) for coord in box]
        
        obj_w = base_box[2] - base_box[0]
        obj_h = base_box[3] - base_box[1]
        obj_longer_side = max(obj_w, obj_h)
        
        max_scale_for_obj = target_size / obj_longer_side if obj_longer_side > 0 else 1.0
        min_scale_for_no_padding = 1.0

        if max_scale_for_obj < min_scale_for_no_padding:
            # Case A: Object is too large.
            image_longer_side = max(base_w, base_h)
            min_scale = target_size / image_longer_side
            random_scale = random.uniform(min_scale, max_scale_for_obj)
        else:
            # Case B: Object fits.
            random_scale = random.uniform(min_scale_for_no_padding, max_scale_for_obj)
            
        final_w = int(base_w * random_scale)
        final_h = int(base_h * random_scale)
        final_resized_image = base_image.resize((final_w, final_h), Image.Resampling.BILINEAR)
        final_box = [int(coord * random_scale) for coord in base_box]

        # --- 4. (NEW) Pre-Crop Padding: Ensure image is large enough for cropping ---
        image_to_crop = final_resized_image
        box_to_crop = final_box

        padded_w = max(final_w, target_size)
        padded_h = max(final_h, target_size)

        if final_w < target_size or final_h < target_size:
            # Create a black canvas of the required padded size
            padded_image = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
            
            # Randomly position the image on the canvas
            paste_x = random.randint(0, padded_w - final_w)
            paste_y = random.randint(0, padded_h - final_h)
            padded_image.paste(final_resized_image, (paste_x, paste_y))
            
            # Update the image and box coordinates for the final crop
            image_to_crop = padded_image
            box_to_crop = [
                final_box[0] + paste_x,
                final_box[1] + paste_y,
                final_box[2] + paste_x,
                final_box[3] + paste_y,
            ]
        
        # --- 5. Final Random Crop ---
        # At this point, image_to_crop is guaranteed to be >= target_size in both dimensions.
        min_left = box_to_crop[2] - target_size
        max_left = box_to_crop[0]
        min_top = box_to_crop[3] - target_size
        max_top = box_to_crop[1]

        valid_min_left = max(0, min_left)
        valid_max_left = min(image_to_crop.width - target_size, max_left)
        valid_min_top = max(0, min_top)
        valid_max_top = min(image_to_crop.height - target_size, max_top)

        left = random.randint(valid_min_left, valid_max_left) if valid_min_left <= valid_max_left else (valid_min_left + valid_max_left) // 2
        top = random.randint(valid_min_top, valid_max_top) if valid_min_top <= valid_max_top else (valid_min_top + valid_max_top) // 2
        
        final_image = Func.crop(image_to_crop, top, left, target_size, target_size)
            
    return final_image


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        fewshot_dir,
        mask_dir,
        tokenizer,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        tokenizer_max_length=None,
        dataset=None,
        n_shot=None,
        n_template=None,
        target_class_idx=None,
        instance_fname=None,
        instance_fname_ext=None,
        train_batch_size=8,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.dataset = dataset
        self.n_shot = n_shot
        self.templates = TEMPLATES_SMALL[:n_template]
        dataset_name = {
            "cifar10": " ",
            "isic": "skin diseases ",
            "imagenet": "",
            "imagenet100": "",
            "fgvc_aircraft": "aircraft ",
            "fgvc_aircraft_novel": "aircraft ",
            "cars": "car ",
            "cars_novel": "car ",
            "eurosat": "satellite ",
            "eurosat_novel": "satellite ",
            "food101": "food ",
            "food101_novel": "food ",
            "dtd": "texture ",
            "dtd_novel": "texture ",
            "pets": "pet ",
            "pets_novel": "pet ",
            "flowers102": "flower ",
            "flowers102_novel": "flower ",
            "sun397": "scene ",
            "caltech101": "",
            "caltech101_novel": "",
        }[dataset]
        self.dataset_name = dataset_name
        self.fewshot_dir = fewshot_dir
        self.target_class_idx = target_class_idx
        self.instance_fname = instance_fname
        self.instance_fname_ext = instance_fname_ext
        self.train_batch_size = train_batch_size

        self.fewshot_dir = Path(fewshot_dir)
        if not self.fewshot_dir.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.exists():
            self.instance_masks_path = []
            self.instance_images_path, self.instance_prompts, self.num_instance_images = self.get_instance_data_list_wo_mask()
        else:
            self.instance_masks_path, self.instance_images_path, self.instance_prompts, self.num_instance_images = self.get_instance_data_list()
        self._length = self.num_instance_images

        tf_crop = (
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        )
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                tf_crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def get_instance_data_list_wo_mask(self):
        instance_images_path = []
        instance_prompts = []

        for cls_idx, classname in enumerate(CLASSNAMES[self.dataset]):
            if self.target_class_idx is not None:
                if cls_idx == self.target_class_idx: pass
                else: continue

            data_dir = ospj(self.fewshot_dir, classname)
            if self.instance_fname is None:
                image_fnames = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=natural_keys)
                images_path = [ospj(data_dir, f) for f in image_fnames]

                if self.n_shot is not None:
                    images_path = images_path[: self.n_shot]
            else:
                image_filename = self.instance_fname + "." + self.instance_fname_ext
                images_path = [ospj(data_dir, image_filename)] * self.train_batch_size

            for _ in range(len(images_path)):
                template = random.choice(self.templates)
                prompt = template.format(self.dataset_name, classname)
                instance_prompts.append(prompt)

            instance_images_path += images_path

        num_instance_images = len(instance_images_path)

        return instance_images_path, instance_prompts, num_instance_images

    def get_instance_data_list(self):
        instance_masks_path = []
        instance_images_path = []
        instance_prompts = []

        for cls_idx, classname in enumerate(CLASSNAMES[self.dataset]):
            if self.target_class_idx is not None:
                if cls_idx == self.target_class_idx: pass
                else: continue

            mask_dir = ospj(self.mask_dir, classname)
            data_dir = ospj(self.fewshot_dir, classname)
            if self.instance_fname is None:
                image_fnames = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], key=natural_keys)
                masks_path = [ospj(mask_dir, Path(f).stem + ".json") for f in image_fnames]
                images_path = [ospj(data_dir, f) for f in image_fnames]

                if self.n_shot is not None:
                    masks_path = masks_path[: self.n_shot]
                    images_path = images_path[: self.n_shot]
            else:
                image_filename = self.instance_fname + "." + self.instance_fname_ext
                mask_filename = self.instance_fname + ".json"
                images_path = [ospj(data_dir, image_filename)] * self.train_batch_size
                masks_path = [ospj(mask_dir, mask_filename)] * self.train_batch_size

            for _ in range(len(images_path)):
                template = random.choice(self.templates)
                prompt = template.format(self.dataset_name, classname)
                instance_prompts.append(prompt)

            instance_masks_path += masks_path
            instance_images_path += images_path

        num_instance_images = len(instance_images_path)

        return instance_masks_path, instance_images_path, instance_prompts, num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_idx = index % self.num_instance_images
        instance_image = Image.open(self.instance_images_path[instance_idx])
        instance_image = exif_transpose(instance_image)
        instance_prompt = self.instance_prompts[instance_idx]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        if self.instance_masks_path != []:
            with open(self.instance_masks_path[instance_idx], 'r', encoding='utf-8') as f:
                data = json.load(f)
                instance_mask = data['enclosing_box'] # [x_min, y_min, x_max, y_max]

            if not box_has_area(instance_mask, instance_image.width, instance_image.height):
                warnings.warn("Invalid bbox; falling back to default transforms.")
                example["instance_images"] = self.image_transforms(instance_image)
            else:
                s, one_side_ok, s_min_used, s_max_used = choose_scale_one_side_pad(
                    instance_image.width, instance_image.height, instance_mask, self.size, rng=random
                )
                cropped_image = crop_centered(
                    image=instance_image,
                    box=instance_mask,
                    S=self.size,
                    s=s,
                    pad_color=(0, 0, 0)
                )
                # cropped_image = augment_image_by_spec(instance_image, instance_mask, self.size)
                final_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                example["instance_images"] = final_transforms(cropped_image)
        else:
            example["instance_images"] = self.image_transforms(instance_image)
        # example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer,
                instance_prompt,
                tokenizer_max_length=self.tokenizer_max_length,
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
        example["adapter_name"] = f"B{index}_each" # check when multiple shots

        return example


def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    adapter_names = [ex["adapter_name"] for ex in examples]
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "adapter_names": adapter_names,  
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(
    text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None
):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    text_encoder.requires_grad_(False)

    # We only train the additional adapter LoRA layers
    if vae is not None:
        vae.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    if args.train_unet:
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        unet.add_adapter(unet_lora_config)
    
    if args.train_unet:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )

        shared_A_name = "A_shared"
        num_classes = len(CLASSNAMES[args.dataset])
        num_instances = num_classes * args.n_shot  # total number of training instances
        b_names = [f"B{i}_each" for i in range(num_instances)]

        unet.add_adapter(lora_config, adapter_name=shared_A_name)
        for name in b_names:
            unet.add_adapter(lora_config, adapter_name=name)

        def tie_lora_A_weights(model, shared_adapter_name, target_adapter_names):
            for m in model.modules():
                if not hasattr(m, "lora_A"):
                    continue

                base_A_module = m.lora_A[shared_adapter_name]
                base_A_weight = base_A_module.weight

                for name in target_adapter_names:
                    m.lora_A[name].weight = base_A_weight

        tie_lora_A_weights(unet, shared_adapter_name=shared_A_name, target_adapter_names=b_names)
        shared_layers = 0
        for module in unet.modules():
            if hasattr(module, "lora_A"):
                base = module.lora_A[shared_A_name].weight
                assert all(module.lora_A[name].weight is base for name in b_names)
                shared_layers += 1
        print("LIVE_A_SHARING_AUDIT", {"layers": shared_layers, "heads": len(b_names),
              "scope": "official_dataset_wide", "same_parameter_object": True}, flush=True)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    active_adapters = b_names
    adapter_weights = [1.0] * len(b_names)
    unet.set_adapters(active_adapters, adapter_weights)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            unet_, unet_state_dict, adapter_name="default"
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.append(text_encoder_)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.append(text_encoder)

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_A_n, params_B_n = [], []
    params_A, params_B = [], []
    for n, p in unet.named_parameters():
        if p.requires_grad:
            if "lora_A.A_shared" in n:
                params_A_n.append(n)
                params_A.append(p)
            else:
                params_B_n.append(n)
                params_B.append(p)

    print("OPTIMIZER_AUDIT", {"A_tensors": len(params_A), "B_tensors": len(params_B),
          "text_encoder_in_optimizer": False}, flush=True)
    optimizer_A = optimizer_class(
        params_A,
        lr=args.learning_rate_A,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_B = optimizer_class(
        params_B,
        lr=args.learning_rate_B,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    pre_computed_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        fewshot_dir=args.fewshot_dir,
        mask_dir=args.mask_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
        dataset=args.dataset,
        n_shot=args.n_shot,
        n_template=args.n_template,
        target_class_idx=args.target_class_idx,
        instance_fname=args.instance_fname,
        instance_fname_ext=args.instance_fname_ext,
        train_batch_size=args.train_batch_size,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )
    

    # save prompts
    fpath = ospj(args.output_dir, "prompts.json")
    with open(fpath, "w") as f:
        json.dump(train_dataset.instance_prompts, f, indent=2)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_A = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_A,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler_B = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_B,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer_A,
            optimizer_B,
            train_dataloader,
            lr_scheduler_A,
            lr_scheduler_B
        ) = accelerator.prepare(
            unet, text_encoder, optimizer_A, optimizer_B, train_dataloader, lr_scheduler_A, lr_scheduler_B
        )
    else:
        unet, optimizer_A, optimizer_B, train_dataloader, lr_scheduler_A, lr_scheduler_B = accelerator.prepare(
            unet, optimizer_A, optimizer_B, train_dataloader, lr_scheduler_A, lr_scheduler_B
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        #         disable=not accelerator.is_local_main_process,
        disable=not args.is_tqdm,
    )

    current = time.time()
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                bsz = batch["pixel_values"].shape[0]
                loss_sum = 0

                if args.method == "JOINT":
                    for i in range(bsz):
                        b_name = batch["adapter_names"][i]
                        unet.set_adapters(b_name, [1.0])

                        pixel_val = batch["pixel_values"][i:i+1].to(dtype=weight_dtype)
                        if vae is not None:
                            model_input = vae.encode(pixel_val).latent_dist.sample() * vae.config.scaling_factor
                        else:
                            model_input = pixel_val

                        noise      = torch.randn_like(model_input)
                        timesteps  = torch.randint(
                            0, 
                            noise_scheduler.config.num_train_timesteps, 
                            (1,),
                            device=model_input.device,
                            dtype=torch.long
                        )
                        noisy      = noise_scheduler.add_noise(model_input, noise, timesteps)

                        enc_hidden = encode_prompt(
                            text_encoder,
                            batch["input_ids"][i:i+1],
                            batch["attention_mask"][i:i+1],
                            text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                        )

                        if unwrap_model(unet).config.in_channels == model_input.shape[1] * 2:
                            noisy = torch.cat([noisy, noisy], dim=1)

                        pred = unet(noisy, timesteps, enc_hidden, return_dict=False)[0]
                        if pred.shape[1] == 6:        
                            pred, _ = torch.chunk(pred, 2, dim=1)

                        target = {
                            "epsilon": noise,
                            "v_prediction": noise_scheduler.get_velocity(model_input, noise, timesteps),
                        }[noise_scheduler.config.prediction_type]

                        loss_sum += F.mse_loss(pred.float(), target.float(), reduction="mean")

                    for p in params_B: p.requires_grad = True
                    for p in params_A: p.requires_grad = True

                    loss = loss_sum / bsz           
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_B, args.max_grad_norm)
                        accelerator.clip_grad_norm_(params_A, args.max_grad_norm)
                    optimizer_B.step()
                    lr_scheduler_B.step()
                    optimizer_B.zero_grad()
                    optimizer_A.step()
                    lr_scheduler_A.step()
                    optimizer_A.zero_grad()
                
                else:
                    if (step // args.step) % 2 == 0:
                        for i in range(bsz):
                            b_name = batch["adapter_names"][i]
                            unet.set_adapters(b_name, [1.0])

                            pixel_val = batch["pixel_values"][i:i+1].to(dtype=weight_dtype)
                            if vae is not None:
                                model_input = vae.encode(pixel_val).latent_dist.sample() * vae.config.scaling_factor
                            else:
                                model_input = pixel_val

                            noise      = torch.randn_like(model_input)
                            timesteps  = torch.randint(
                                0, 
                                noise_scheduler.config.num_train_timesteps, 
                                (1,),
                                device=model_input.device,
                                dtype=torch.long
                            )
                            noisy      = noise_scheduler.add_noise(model_input, noise, timesteps)

                            enc_hidden = encode_prompt(
                                text_encoder,
                                batch["input_ids"][i:i+1],
                                batch["attention_mask"][i:i+1],
                                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                            )

                            if unwrap_model(unet).config.in_channels == model_input.shape[1] * 2:
                                noisy = torch.cat([noisy, noisy], dim=1)

                            pred = unet(noisy, timesteps, enc_hidden, return_dict=False)[0]
                            if pred.shape[1] == 6:        
                                pred, _ = torch.chunk(pred, 2, dim=1)

                            target = {
                                "epsilon": noise,
                                "v_prediction": noise_scheduler.get_velocity(model_input, noise, timesteps),
                            }[noise_scheduler.config.prediction_type]

                            loss_sum += F.mse_loss(pred.float(), target.float(), reduction="mean")

                        for p in params_B: p.requires_grad = True
                        for p in params_A: p.requires_grad = False

                        loss = loss_sum / bsz           
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(params_B, args.max_grad_norm)
                        optimizer_B.step()
                        lr_scheduler_B.step()
                        optimizer_B.zero_grad()
                    else:
                        for i in range(bsz):
                            b_name = batch["adapter_names"][i]
                            unet.set_adapters(b_name, [1.0])

                            pixel_val = batch["pixel_values"][i:i+1].to(dtype=weight_dtype)
                            if vae is not None:
                                model_input = vae.encode(pixel_val).latent_dist.sample() * vae.config.scaling_factor
                            else:
                                model_input = pixel_val

                            noise      = torch.randn_like(model_input)
                            timesteps  = torch.randint(
                                0, 
                                noise_scheduler.config.num_train_timesteps, 
                                (1,),
                                device=model_input.device,
                                dtype=torch.long
                            )
                            noisy      = noise_scheduler.add_noise(model_input, noise, timesteps)

                            enc_hidden = encode_prompt(
                                text_encoder,
                                batch["input_ids"][i:i+1],
                                batch["attention_mask"][i:i+1],
                                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                            )

                            if unwrap_model(unet).config.in_channels == model_input.shape[1] * 2:
                                noisy = torch.cat([noisy, noisy], dim=1)

                            pred = unet(noisy, timesteps, enc_hidden, return_dict=False)[0]
                            if pred.shape[1] == 6:        
                                pred, _ = torch.chunk(pred, 2, dim=1)

                            target = {
                                "epsilon": noise,
                                "v_prediction": noise_scheduler.get_velocity(model_input, noise, timesteps),
                            }[noise_scheduler.config.prediction_type]

                            loss_sum += F.mse_loss(pred.float(), target.float(), reduction="mean")

                        for p in params_B: p.requires_grad = False
                        for p in params_A: p.requires_grad = True

                        loss = loss_sum / bsz           
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(params_A, args.max_grad_norm)
                        optimizer_A.step()
                        lr_scheduler_A.step()
                        optimizer_A.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            if accelerator.sync_gradients and (global_step == 1 or global_step % 50 == 0):
                progress = dict(epoch=epoch + 1, epochs=args.num_train_epochs,
                                step=global_step, total_steps=args.max_train_steps,
                                loss=logs["loss"], elapsed_seconds=time.time() - current,
                                peak_gpu_bytes=torch.cuda.max_memory_allocated())
                print("TRAIN_PROGRESS", json.dumps(progress), flush=True)
                with open(ospj(args.output_dir, "training_progress.json"), "w") as f:
                    json.dump(progress, f, indent=2)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (epoch == args.num_train_epochs - 1):
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # Save the lora layers
                accelerator.wait_for_everyone()

                # save LoRA
                unet = unwrap_model(unet)
                unet = unet.to(torch.float32)

                if args.train_text_encoder:
                    text_encoder = unwrap_model(text_encoder)
                    text_encoder_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(text_encoder)
                    )
                else:
                    text_encoder_state_dict = None

                unet_lora_state_dicts = {}
                if args.train_unet:
                    shared_A_sd = get_peft_model_state_dict(unet, adapter_name=shared_A_name)
                    
                    for name in b_names:
                        clean_sd = {}
                        # Add the shared A weights to the clean state dict
                        for k, v in shared_A_sd.items():
                            if "lora_A" in k:
                                # Rename key to be generic for saving
                                new_key = k.replace(f".{shared_A_name}.", ".default.")
                                clean_sd[new_key] = v
                        
                        # Get the specific B weights
                        b_sd = get_peft_model_state_dict(unet, adapter_name=name)
                        for k, v in b_sd.items():
                            if "lora_B" in k:
                                # Rename key to be generic for saving
                                new_key = k.replace(f".{name}.", ".default.")
                                clean_sd[new_key] = v
                        unet_lora_state_dicts[name] = convert_state_dict_to_diffusers(clean_sd)

                if args.train_unet or args.train_text_encoder:
                    classnames = CLASSNAMES[args.dataset]
                    for idx, name in enumerate(b_names):
                        # Map global B index to class and local index
                        class_idx = idx // args.n_shot
                        local_idx = idx % args.n_shot
                        classname = classnames[class_idx]
                        local_name = f"B{local_idx}_each"

                        adapter_save_dir = os.path.join(args.output_dir, classname, local_name)
                        os.makedirs(adapter_save_dir, exist_ok=True)

                        unet_layers_to_save = unet_lora_state_dicts.get(name)

                        LoraLoaderMixin.save_lora_weights(
                            save_directory=adapter_save_dir,
                            unet_lora_layers=unet_layers_to_save,
                            text_encoder_lora_layers=text_encoder_state_dict,
                        )

                del pipeline
                torch.cuda.empty_cache()

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Time:", time.time() - current)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
