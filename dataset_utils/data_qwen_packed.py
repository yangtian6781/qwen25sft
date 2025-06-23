import os
import copy
import json
import random
import logging
import sys
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from collections.abc import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder

import transformers
from .rope2d import get_rope_index_25
import math

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"


FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

local_rank = None



def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []


    try:
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]
    except:
        logger.info(source)

    input_id, target = [], []

    input_id += tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message}]
    )
    target += [IGNORE_INDEX] * len(input_id)

    for conv in source:
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        role = roles.get(role, role)
        if role == "user":
            if DEFAULT_IMAGE_TOKEN in content:
                parts = content.split(DEFAULT_IMAGE_TOKEN)
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|image_pad|>"
                        * grid_thw_image[visual_replicate_index_image]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                    visual_replicate_index_image += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            if DEFAULT_VIDEO_TOKEN in content:
                parts = content.split(DEFAULT_VIDEO_TOKEN)
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|video_pad|>"
                        * grid_thw_video[visual_replicate_index_video]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                    visual_replicate_index_video += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)

        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        if role in ["user", "system"]:
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            target_mask = encode_id.copy()
            target_mask[:3] = [IGNORE_INDEX] * 3
            target += target_mask

    assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
    input_ids.append(input_id)
    targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class Qwen25LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset, image_processor, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(Qwen25LazySupervisedDataset, self).__init__()

        self.ds_name = dataset
        self.root = dataset["root"]
        self.get_rope_index = get_rope_index_25
        file_format = dataset["annotation"].split(".")[-1]
        if file_format == "jsonl":
            self.list_data_dict = read_jsonl(dataset["annotation"])
        else:
            self.list_data_dict = json.load(open(dataset["annotation"], "r"))
        repeat_time = dataset.get("repeat_time", 1)
        if repeat_time < 1:
            # If repeat_time is less than 1, select a portion of the data
            self.list_data_dict = self.list_data_dict[:int(len(self.list_data_dict) * repeat_time)]
        if repeat_time > 1:
            assert isinstance(repeat_time, int)
            # Repeat the list if repeat_time is greater than 1
            self.list_data_dict = self.list_data_dict * repeat_time


        logger.info(f"Total training samples: {len(self.list_data_dict)}")
        logger.info("Formatting inputs...Skip in lazy mode")
        self.tokenizer = copy.deepcopy(tokenizer)
        self.image_processor = copy.deepcopy(image_processor)

        self.data_args = data_args
        self.max_token_length = getattr(self.data_args, "max_seq_length", 4096)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            logger.info("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.image_processor)
        processor.max_pixels = self.data_args.image_max_pixels
        processor.min_pixels = self.data_args.image_min_pixels
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
            except Exception as e:
                logger.info(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            logger.info(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            logger.info(f"File not exist: {video_file}")
            return None
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        if self.data_args.nframes is not None:
            target_frames = round_by_factor(min(self.data_args.nframes, target_frames), FRAME_FACTOR)
        else:
            assert self.data_args.fps is not None
            fps = self.data_args.fps
            min_frames = ceil_by_factor(self.data_args.get("video_min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(self.data_args.get("video_max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
            nframes = total_frames / avg_fps * fps
            if nframes > total_frames:
                logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
            nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
            nframes = floor_by_factor(nframes, FRAME_FACTOR)
        # interval = getattr(self.data_args, "base_interval", 4)

        # num_frames_to_sample = round(video_length / interval)
        # video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        # video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        # target_frames = min(
        #     max(num_frames_to_sample, video_min_frames), video_max_frames
        # )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = self.list_data_dict[i]
                ret = self.get_data(data_item)
                break
            except Exception as e:
                try_cnt += 1
                logger.exception(e)
                logger.exception(self.ds_name)
                i = random.randint(0, len(self.list_data_dict) - 1)
        return ret


    def get_data(self, sources) -> Dict[str, torch.Tensor]:
        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources:
            image_folder = self.root
            image_file = sources["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources:
            video_file = self.root
            video_folder = sources["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy(sources["conversations"])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["input_ids"] = data_dict["input_ids"][..., :self.max_token_length]
        data_dict["labels"] = data_dict["labels"][..., :self.max_token_length]
        if data_dict["labels"].max() == -100:
            raise RuntimeError("max_seq_length can not capture valid token to calculate loss")
        data_dict["position_ids"] = data_dict["position_ids"][..., :self.max_token_length]
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        if "image" in sources:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in sources:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        return data_dict



def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenDataCollatorForSupervisedDataset(object):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch



def build_datasets(
    meta_path,
    image_processor,
    tokenizer,
    data_args
):
    datasets = []
    ds_collections = json.loads(open(meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        dataset = Qwen25LazySupervisedDataset(
            ds_collections[ds_name],
            image_processor=image_processor,
            tokenizer=tokenizer,
            data_args=data_args
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)

    train_dataset = ConcatDataset(datasets)
    return train_dataset





if __name__ == "__main__":
    debug= {
        "root": "/share/tianyang/data/debug",
        "annotation": "/share/tianyang/data/debug/train.jsonl",
        "repeat_time": 1
    }
    from transformers import Qwen2_5_VLProcessor
    qwen25_processor = Qwen2_5_VLProcessor.from_pretrained("/share/tianyang/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)
    dataset = Qwen25LazySupervisedDataset(dataset=debug,
                                          image_processor=qwen25_processor.image_processor,
                                          tokenizer=qwen25_processor.tokenizer,
                                          data_args={})
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=DataCollatorForSupervisedDataset(tokenizer=qwen25_processor.tokenizer))
    for i in dataloader:
        a = 1
