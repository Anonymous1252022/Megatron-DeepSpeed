#!/bin/bash

# --- for debug only ---
export DEBUG="" # "true"
# --- for debug only ---

# fp8_linear
export HL_FP8_LINEAR="true"
export HL_DISABLE_FP8="false" # "true" # For debug only !
export HL_DISABLE_FP8_MLP_LINEAR2="false" # "true"
export HL_SCALED_SWIGLU="true"
export HL_DETACH_SCALED_SWIGLU="true"
export HL_DELAYED_SCALED_SWIGLU="false" # "true"

# fp8_optimizer (simulation)
export HL_FP8_OPTIMIZER="true"
export HL_MW_DTYPE="fp16_pts" # "fp32", "bf16", "bf16_sr", "fp16", "fp16_pts", "fp8_e4m3", "fp8_e4m3_sr", "fp8_e5m2", "fp8_e5m2_sr"
export HL_M1_DTYPE="fp8_e4m3" # "fp32", "bf16", "bf16_sr", "fp16", "fp16_pts", "fp8_e4m3", "fp8_e4m3_sr", "fp8_e5m2", "fp8_e5m2_sr"
export HL_M2_DTYPE="fp8_e5m2" # "fp32", "bf16", "bf16_sr", "fp16", "fp16_pts", "fp8_e4m3", "fp8_e4m3_sr", "fp8_e5m2", "fp8_e5m2_sr"
export HL_SFTZ="false" # "true"

# other params
export HL_DATA_DIR_ROOT="/mnt/weka/algo/red_pajama"
export HL_DATA_CACHE_DIR="/software/data/dataset_idx/red_pajama"
export HL_TOKENIZER_MODEL="/software/data/datasets/red_pajama/tokenizer.model"
export HL_DEVICES_PER_NODE=8
export HL_DP=8
export HL_TP=1
export HL_PP=1
export HL_MICRO_BATCH=1
export HL_EXIT_INTERVAL=0
export HL_SAVE_INTERVAL=1000
export HL_CKP_ACT=2
export HL_LLAMA_VER=2
export HL_LLAMA_MODEL_SIZE=7
export HL_DROPOUT=0
export HL_SEQ_PARALLEL=0
export HL_USE_FUSED_RMSNORM=1
export HL_OPTIMIZER="fusedadamw"
export HL_EVAL_ITERS=100
export HL_EVAL_INTERVAL=3000
export HL_HOSTSFILE="/root/hostsfile"
export HL_RESULTS_DIR="/root/output/test"
export HL_SAVE=0
export HL_NUM_LAYERS=32
export HL_NUM_HEADS=32
export HL_LOG_INTERVAL=1
export HL_CHECKPOINTS_DIR=""
export HL_UNIV_CP=0
export HL_USE_FAST_SOFTMAX=1

./run_llama.sh
