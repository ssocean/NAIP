module load compiler/gcc/gcc-7.5.0-gcc-4.8.5-4qwzk6c

source ~/miniconda3/etc/profile.d/conda.sh
conda activate naipv2

which python
python --version

python test.py \
  --ckpt_dir "$CKPT_DIR" \
  --test_data_path "$TEST_SET" \
  --gt_field "RTS" \
  --batch_size 8 \
  --max_length 512 \
  --load_in_8bit true
