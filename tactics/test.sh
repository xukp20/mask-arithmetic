PROJECT_BASE=/cephfs/xukangping/code/mask-arithmetic
MODEL_PATH=$PROJECT_BASE/tactics/outputs/position
EVAL_FILE=$PROJECT_BASE/tactics/data_sets/data/eval_baseline_position.jsonl
EVAL_SIZE=500
TOKENIZER_PATH=$PROJECT_BASE/tactics/data_sets/data/6_tactics_tokenizer

cd ../extrapolate

python test.py --model_path $MODEL_PATH --eval_file $EVAL_FILE --eval_size $EVAL_SIZE --tokenizer_path $TOKENIZER_PATH