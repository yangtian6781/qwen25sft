import logging
import sys
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional
import tqdm
from peft import LoraConfig, get_peft_model

import torch
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset, DataLoader
from transformers import HfArgumentParser, AutoTokenizer
import transformers
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from model_utils.qwen25_flatten import replace_qwen25_vl_attention_class
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DataLoaderConfiguration
from accelerate.utils.random import set_seed
from dataset_utils.data_qwen_packed import build_datasets, DataCollatorForSupervisedDataset, FlattenDataCollatorForSupervisedDataset

from torch.optim import AdamW
from timm.scheduler import CosineLRScheduler
from deepspeed import DeepSpeedEngine
# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    shuffle: Optional[bool] = field(
        default=True, metadata={'help': 'shuffle'}
    )
    meta_path: Optional[str] = field(
        default="./shell/data/use_gpu.json",
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    nframes: Optional[int] = field(
        default=32, metadata={'help': 'nframes'}
    )
    fps: Optional[float] = field(
        default=2.0, metadata={'help': 'fps'}
    )
    video_min_frames: Optional[int] = field(
        default=4, metadata={'help': 'video_min_frames'}
    )
    video_max_frames: Optional[int] = field(
        default=32, metadata={'help': 'video_max_frames'}
    )
    video_min_frame_pixels: Optional[int] = field(
        default=448*448, metadata={'help': 'video_min_frame_pixels'}
    )
    video_max_frame_pixels: Optional[int] = field(
        default=1024*1024, metadata={'help': 'video_max_frame_pixels'}
    )
    image_min_pixels: Optional[int] = field(
        default=448*448, metadata={'help': 'image_min_pixels'}
    )
    image_max_pixels: Optional[int] = field(
        default=1792*1792, metadata={'help': 'image_max_pixels'}
    )


@dataclass
class TrainingArguments:
    project_dir: str = field(
        default= None,
        metadata={"help": "project dir"},
    )
    exp_dir: str = field(
        default= None,
        metadata={"help": "experiment dir"},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU for training."}
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    pack_data: bool = field(default=False, metadata={"help": "yes or no pack data to accelerate training."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    peak_lr: float = field(default=5e-5, metadata={"help": "Peak Lr for AdamW optimizer."})
    warmup_lr_init: float = field(default=1e-7, metadata={"help": "The initial warm up learning rate for AdamW."})
    lr_min: float = field(default=1e-7, metadata={"help": "min lr."})
    warmup_step: float = field(default=0.1, metadata={"help": "warm up steps."})
    weight_decay: float = field(default=0.05, metadata={"help": "Weight Decay."})
    logging_steps: int = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer."
            )
        },
    )
    seed: int = field(default=42, metadata={"help": "random seed"})
    vscode_debug: bool = field(default=False, metadata={"help": "vscode debug mode"})


def replace_qwenrmsnorm_with_apex():
    from apex.normalization import FusedRMSNorm
    from functools import partial
    qwen_fused_rmsnorm = partial(FusedRMSNorm, eps=1e-6)
    transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm = qwen_fused_rmsnorm


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    qwen25_processor = Qwen2_5_VLProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True
    )
    image_processor = qwen25_processor.image_processor
    tokenizer = qwen25_processor.tokenizer
    replace_qwenrmsnorm_with_apex()
    apply_liger_kernel_to_qwen2_5_vl(rms_norm=False)
    if training_args.pack_data:
        data_collator = FlattenDataCollatorForSupervisedDataset(tokenizer)
        replace_qwen25_vl_attention_class()
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer)
    
    project_dir = training_args.project_dir
    exp_dir = training_args.exp_dir
    project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=exp_dir)
    accelerator = Accelerator(log_with='tensorboard', 
                              project_config=project_config, 
                              dataloader_config=DataLoaderConfiguration(non_blocking=True))
    accelerator.init_trackers(project_name='logging')
    if training_args.vscode_debug and accelerator.is_main_process:
        import debugpy         
        debugpy.listen(('127.0.0.1', 7389))
        print("waiting for debug", flush=True)
        debugpy.wait_for_client()
    set_seed(training_args.seed)

    # Log on each process the small summary:
    if accelerator.is_local_main_process:
        logger.info(
            f'Process rank: {accelerator.process_index}, device: {accelerator.device}, n_gpu: {accelerator.num_processes}'
            + f'distributed training: {bool(accelerator.process_index != -1)}, 16-bit training: {accelerator.mixed_precision}'
        )
        logger.info(f'Model parameters\n {model_args}')
        logger.info(f'Data parameters\n {data_args}')
        logger.info(f'Training parameters\n {training_args}')


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    logger.info('Finished')
    if model_args.grad_checkpoint:
        model.gradient_checkpointing_enable()



    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False


    if model_args.freeze_backbone:
        _freeze_params(model.visual)

    if model_args.freeze_llm:
        _freeze_params(model.model)

    if model_args.unfreeze_lm_head:
        model.lm_head.weight.requires_grad = True

    if model_args.use_backbone_lora:
        def get_input_embeddings(module):
            return module.proj
        from functools import partial
        
        model.visual.base_model_prefix = "patch_embed"
        model.visual.get_input_embeddings = partial(get_input_embeddings, module=model.visual.patch_embed)
        loraconfig = LoraConfig(
            r=model_args.use_backbone_lora,
            target_modules=['qkv', 'proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_alpha=2 * model_args.use_backbone_lora,
            lora_dropout=0.05,
        )
        model.visual = get_peft_model(model.visual, loraconfig)
        model.visual.print_trainable_parameters()

    if model_args.use_llm_lora:
        loraconfig = LoraConfig(
            r=model_args.use_llm_lora,
            target_modules="all-linear",
            lora_alpha=2 * model_args.use_llm_lora,
            lora_dropout=0.05,
        )
        model.model = get_peft_model(model.model, loraconfig)
        model.model.print_trainable_parameters()

    if model_args.freeze_mlp:
        _freeze_params(model.visual.merger)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.visual.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True
    model.train()
    # print trainable parameters
    if accelerator.is_local_main_process:
        trainable_param = []
        untrainable_param = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param.append(name)
            else:
                untrainable_param.append(name)
        for i in trainable_param:
            logger.info(f'trainable parameter:   {i}')
        for i in untrainable_param:
            logger.info(f'untrainable parameter:   {i}')
    torch.distributed.barrier()
    

    train_dataset = build_datasets(meta_path=data_args.meta_path,
                                   image_processor=image_processor,
                                   tokenizer=tokenizer,
                                   data_args=data_args)

    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=training_args.per_device_train_batch_size,
                            shuffle=data_args.shuffle,
                            num_workers=training_args.dataloader_num_workers,
                            collate_fn=data_collator,
                            drop_last=False,
                            pin_memory=True)
    weight_param_groups = []
    bias_param_groups = []
    for name, value in model.named_parameters():
        if 'bias' in name and value.requires_grad != False:
            bias_param_groups.append(value)
        if 'bias' not in name and value.requires_grad != False:
            weight_param_groups.append(value)
    param_groups = [{'params': weight_param_groups, 'weight_decay': training_args.weight_decay},
                    {'params': bias_param_groups, 'weight_decay': 0}]
    optimizer = AdamW(params=param_groups,
                      lr=training_args.peak_lr,
                      betas=(training_args.adam_beta1, training_args.adam_beta2),
                      eps=training_args.adam_epsilon,
                      weight_decay=0)

    complete_segment = len(dataloader) % (base := accelerator.gradient_accumulation_steps * accelerator.num_processes) == 0
    total_step = len(dataloader) // base if complete_segment else len(dataloader) // base + 1
    warmup_step = int(training_args.warmup_step) if training_args.warmup_step>1 else int(total_step*training_args.warmup_step)
    lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                     warmup_t=warmup_step,
                                     warmup_lr_init=training_args.warmup_lr_init,
                                     warmup_prefix=True,
                                     t_initial=total_step-warmup_step,
                                     lr_min=training_args.lr_min)

    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)
    model:DeepSpeedEngine
    accelerator.register_for_checkpointing(lr_scheduler)

    log_per_steps = training_args.logging_steps
    
    accelerator.print('----------------------optimizer config-----------------------------')
    accelerator.print(optimizer)
    accelerator.print('----------------------optimizer config-----------------------------')
    accelerator.print('gradient_clipping ', model.gradient_clipping())
    accelerator.print('gradient_accumulation_steps  ', accelerator.gradient_accumulation_steps)
    accelerator.print('model.get_lr(): ', model.get_lr())


    for data in tqdm.tqdm(dataloader, disable=not accelerator.is_local_main_process):
        if accelerator.gradient_state.end_of_dataloader:
            model.set_gradient_accumulation_boundary(is_boundary=True)
        output = model(**data, use_cache=False)
        loss = output.loss
        model.backward(loss)
        if model.is_gradient_accumulation_boundary():
            lr_scheduler.step(model.global_steps)
            deepspeed_loss = model.losses.clone()
            model.step()
            if model.global_steps%log_per_steps == 0 or accelerator.gradient_state.end_of_dataloader:
                torch.distributed.all_reduce(deepspeed_loss, torch.distributed.ReduceOp.AVG)
                lr = model.get_lr()[0]
                accelerator.log(values={'loss': deepspeed_loss.item(),
                                        'grad_norm': model.get_global_grad_norm(),
                                        'lr': lr},
                                step=model.global_steps)
        else:
            model.step()

    torch.distributed.barrier()
    tokenizer.save_pretrained(os.path.join(accelerator.logging_dir, f'finish_train'),
                              is_main_process=accelerator.is_main_process)
    unwrap_model = accelerator.unwrap_model(model)
    if accelerator.deepspeed_config["zero_optimization"]["stage"] == 3:
        if model_args.use_backbone_lora or model_args.use_llm_lora:
            logger.warning("you must merge lora manually!!!!!!!!!!!!!!")
        model.save_pretrained(os.path.join(accelerator.logging_dir, f'finish_train'),
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                    state_dict=accelerator.get_state_dict(model),
                                    safe_serialization=False)
    else:
        if model_args.use_backbone_lora:
            unwrap_model.visual.merge_and_unload()
            unwrap_model.visual = unwrap_model.visual.model
        if model_args.use_llm_lora:
            unwrap_model.model.merge_and_unload()
            unwrap_model.model = unwrap_model.model.model
        unwrap_model.save_pretrained(os.path.join(accelerator.logging_dir, f'finish_train'),
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                    state_dict=accelerator.get_state_dict(unwrap_model))
    torch.distributed.barrier()
    accelerator.end_training()


if __name__ == '__main__':
    main()
    