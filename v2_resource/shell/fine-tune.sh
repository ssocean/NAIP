module load compiler/gcc/gcc-7.5.0-gcc-4.8.5-4qwzk6c

source ~/miniconda3/etc/profile.d/conda.sh


conda activate naipv2


which python
python --version



TOTAL_EPOCHS=1
BATCH_SIZE=8
MAX_PAIRS=10000
MAX_LENGTH=512


# LoRA 固定为你当前最优
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,v_proj"

BASE_ARGS="--total_epochs ${TOTAL_EPOCHS} \
           --batch_size ${BATCH_SIZE} \
           --max_pairs ${MAX_PAIRS} \
           --data_path ${TRAIN_SET} \
           --checkpoint ${CKPT_PTH} \
           --loss_func ${LOSS_FUNC} \
           --max_length ${MAX_LENGTH} \
           --lora_r ${LORA_R} \
           --lora_alpha ${LORA_ALPHA} \
           --lora_dropout ${LORA_DROPOUT} \
           --target_modules ${TARGET_MODULES}"


DATE_STR=$(date +%m%d)
RUN_IDX=0


RUN_DIR="${WORK_DIR}/${DATE_STR}_FINAL"
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/run_train.sh"
PORT=$((10000 + RANDOM % 50000))


OMP_NUM_THREADS=1 accelerate launch --main_process_port=${PORT} "${PY_MAIN}" \
  ${BASE_ARGS} \
  --runs_dir "${RUN_DIR}"
