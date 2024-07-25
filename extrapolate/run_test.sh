# MODEL_PATH=/cephfs/xukangping/code/mask-arithmetic/extrapolate/output/def_model_2024-07-25-16-28-44/checkpoint-40000
# def model
# MODEL_PATH=/cephfs/xukangping/code/mask-arithmetic/extrapolate/output/def_model_2024-07-25-14-16-16
# equal model
MODEL_PATH=/cephfs/xukangping/code/mask-arithmetic/extrapolate/output/def_model_2024-07-25-17-19-27

# IN-DISTRIBUTION
# EVAL_FILE=/cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/eval_def_0-12_0-11_add\&sub_2.jsonl
# EVAL_FILE=/cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/train_def_0-12_0-11_add\&sub_2.jsonl
# EVAL_FILE=/cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/eval_equal_0-11_add\&sub_2_2_2_-1_any.jsonl

# OOD
# EVAL_FILE=/cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/eval_equal_0-12_add\&sub_2_2_2_-1_11.jsonl
EVAL_FILE=/cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/eval_equal_0-12_add\&sub_2_2_2_num-1_tarany_has11.jsonl

EVAL_SIZE=200

python test.py --model_path $MODEL_PATH --eval_file $EVAL_FILE --eval_size $EVAL_SIZE