#!/bin/sh
PYTHON_FILE="{main.py dir}"
DEVICE_IDX="{GPU device number}"
DATA_DIR="{dataset dir}"
OUTPUT_DIR="{save checkpoints}"
RUNFILE_DIR="{save tensorboard logs}"
DATASET="{dataset}"
MODEL_TYPE="{model types}"
DENSE_DIM="{Hidden dimension for encoder and decoder}"
LATENT_DIM="{latent dimension}"
SPLIT_RATIO="0.0"
TRAIN_BATCH_SIZE="256"
TEST_BATCH_SIZE="256"
NUM_EPOCH="{epochs}"
SAVE_STEPS="{stpes}"
OPTIMIZER="adam"
SEED="{seed}"
LEARNING_RATE="4e-4"
WEIGHT_DECAY="{weight_decay}"
ALPHA="1.0"
BETA="{beta}"
GAMMA="1.0"
SUB_SPACE_SIZES_LS="10"
SUB_GROUP_SIZES_LS="100"
REC="{group reconst}"
SUB_LEARNING_RATE="4e-4"
NUM_INV_EQU="{number of IE-transformation function}"

CUDA_VISIBLE_DEVICES=$DEVICE_IDX python $PYTHON_FILE \
--device_idx $DEVICE_IDX \
--data_dir $DATA_DIR \
--output_dir $OUTPUT_DIR \
--run_file $RUNFILE_DIR \
--dataset $DATASET \
--model_type $MODEL_TYPE \
--dense_dim $DENSE_DIM \
--latent_dim $LATENT_DIM \
--split $SPLIT_RATIO \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--test_batch_size $TEST_BATCH_SIZE \
--num_epoch $NUM_EPOCH \
--save_steps $SAVE_STEPS \
--optimizer $OPTIMIZER \
--seed $SEED \
--lr_rate $LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--alpha $ALPHA \
--beta $BETA \
--gamma $GAMMA \
--subgroup_sizes_ls $SUB_GROUP_SIZES_LS \
--subspace_sizes_ls $SUB_SPACE_SIZES_LS \
--hy_rec $REC \
--sub_lr_rate $SUB_LEARNING_RATE \
--num_inv_equ $NUM_INV_EQU \
--do_train --do_eval --write
