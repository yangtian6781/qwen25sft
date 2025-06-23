set -xeuo pipefail

cd /share/tianyang/qwen25vl
NUM_MACHINES=${VC_WORKER_NUM:-1}
MA_NUM_GPUS=${MA_NUM_GPUS:-8}
NUM_PROCESSES=$((MA_NUM_GPUS * NUM_MACHINES))
MACHINE_RANK=${VC_TASK_INDEX:-0}
MAIN_PROCESS_IP=${VC_WORKER_HOSTS%%,*}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

export NCCL_DEBUG=WARN
export NCCL_ALGO=Tree
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=2


work_dirs=${work_dirs:-work_dirs}
project_name=${project_name:-debug}
exp_name=${exp_name:-exp2}
project_dir=${work_dirs}/${project_name}
exp_dir=${project_dir}/${exp_name}
if [ ! -d "$exp_dir" ]; then
  mkdir -p "$exp_dir"
fi

exec > >(tee -a ${exp_dir}/training_log_machine${MACHINE_RANK}.txt) 2>&1

ZERO_STAGE=${ZERO_STAGE:-1}
GRADIENT_CLIPPING=${GRADIENT_CLIPPING:-1.0}
OFFLOAD_OPTIMIZER_DEVICE=${OFFLOAD_OPTIMIZER_DEVICE:-none}
OFFLOAD_PARAM_DEVICE=${OFFLOAD_PARAM_DEVICE:-none}
OFFLOAD_OPTIMIZER_NVME_PATH=${OFFLOAD_OPTIMIZER_NVME_PATH:-/cache}
OFFLOAD_PARAM_NVME_PATH=${OFFLOAD_PARAM_NVME_PATH:-/cache}

BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
shuffle=${shuffle:-true}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NUM_PROCESSES))


video_min_frame_pixels_=$((448 * 448))
video_max_frame_pixels_=$((1024 * 1024))
image_min_pixels_=$((448 * 448))
image_max_pixels_=$((1792 * 1792))


meta_path=${meta_path:-./shell/test.json}
max_seq_length=${max_seq_length:-12000}
pack_data=${pack_data:-true}
nframes=${nframes:-32}
fps=${fps:-2.0}
video_min_frames=${video_min_frames:-4}
video_max_frames=${video_max_frames:-32}
video_min_frame_pixels=${video_min_frame_pixels:-$video_min_frame_pixels_}
video_max_frame_pixels=${video_max_frame_pixels:-$video_max_frame_pixels_}
image_min_pixels=${image_min_pixels:-$image_min_pixels_}
image_max_pixels=${image_max_pixels:-$image_max_pixels_}


grad_checkpoint=${grad_checkpoint:-true}
freeze_backbone=${freeze_backbone:-false}
freeze_mlp=${freeze_mlp:-false}
freeze_llm=${freeze_llm:-false}
unfreeze_lm_head=${unfreeze_lm_head:-true}
unfreeze_vit_layers=${unfreeze_vit_layers:-0}
use_backbone_lora=${use_backbone_lora:-0}
use_llm_lora=${use_llm_lora:-0}
dataloader_num_workers=${dataloader_num_workers:-12}
adam_beta1=${adam_beta1:-0.9}
adam_beta2=${adam_beta2:-0.95}
adam_epsilon=${adam_epsilon:-1e-8}
warmup_step=${warmup_step:-0.1}
warmup_lr_init=${warmup_lr_init:-1e-5}
peak_lr=${peak_lr:-5e-5}
lr_min=${lr_min:-1e-5}
logging_steps=${logging_steps:-1}
seed=${seed:-6781}

vscode_debug=${vscode_debug:-false}

accelerate launch\
  --dynamo_backend no\
  --mixed_precision bf16\
  --num_processes ${NUM_PROCESSES}\
  --num_machines ${NUM_MACHINES}\
  --enable_cpu_affinity\
  --use_deepspeed\
  --same_network\
  --machine_rank ${MACHINE_RANK}\
  --main_process_ip ${MAIN_PROCESS_IP}\
  --main_process_port ${MAIN_PROCESS_PORT}\
  --zero_stage ${ZERO_STAGE}\
  --gradient_clipping ${GRADIENT_CLIPPING}\
  --gradient_accumulation_steps ${GRADIENT_ACC}\
  --offload_optimizer_device ${OFFLOAD_OPTIMIZER_DEVICE}\
  --offload_param_device ${OFFLOAD_PARAM_DEVICE}\
  --offload_optimizer_nvme_path ${OFFLOAD_OPTIMIZER_NVME_PATH}\
  --offload_param_nvme_path ${OFFLOAD_PARAM_NVME_PATH}\
  --zero3_init_flag false\
  --zero3_save_16bit_model true\
  --deepspeed_multinode_launcher standard\
  train.py\
  --model_name_or_path /share/tianyang/huggingface_model/Qwen/Qwen2.5-VL-7B-Instruct\
  --project_dir ${project_dir}\
  --exp_dir ${exp_dir}\
  --meta_path ${meta_path}\
  --max_seq_length ${max_seq_length}\
  --shuffle ${shuffle}\
  --pack_data ${pack_data}\
  --nframes ${nframes}\
  --fps ${fps} \
  --video_min_frames ${video_min_frames}\
  --video_max_frames ${video_max_frames}\
  --video_min_frame_pixels ${video_min_frame_pixels}\
  --video_max_frame_pixels ${video_max_frame_pixels}\
  --image_min_pixels ${image_min_pixels}\
  --image_max_pixels ${image_max_pixels}\
  --grad_checkpoint ${grad_checkpoint}\
  --freeze_backbone ${freeze_backbone}\
  --freeze_mlp ${freeze_mlp}\
  --freeze_llm ${freeze_llm}\
  --unfreeze_lm_head ${unfreeze_lm_head}\
  --unfreeze_vit_layers ${unfreeze_vit_layers}\
  --use_backbone_lora ${use_backbone_lora}\
  --use_llm_lora ${use_llm_lora}\
  --dataloader_num_workers ${dataloader_num_workers}\
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE}\
  --adam_beta1 ${adam_beta1}\
  --adam_beta2 ${adam_beta2}\
  --adam_epsilon ${adam_epsilon}\
  --warmup_step ${warmup_step}\
  --warmup_lr_init ${warmup_lr_init}\
  --peak_lr ${peak_lr}\
  --lr_min ${lr_min}\
  --logging_steps ${logging_steps}\
  --seed ${seed}\
  --vscode_debug ${vscode_debug}