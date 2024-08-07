PROJECT_BASE=/cephfs/xukangping/code/mask-arithmetic
TASK=tactic
MODEL_PATH=$PROJECT_BASE/tactics/outputs/${TASK}_0-11_1-5
# EVAL_FILE=$PROJECT_BASE/tactics/data_sets/data/0-11/eval_500_1-5_all_${TASK}.jsonl
EVAL_FILES=(
    $PROJECT_BASE/tactics/data_sets/data/0-10/eval_500_1-5_all_${TASK}.jsonl
    $PROJECT_BASE/tactics/data_sets/data/0-10/eval_500_10-15_all_${TASK}.jsonl
    $PROJECT_BASE/tactics/data_sets/data/0-11/eval_500_1-5_all_${TASK}.jsonl
    $PROJECT_BASE/tactics/data_sets/data/0-11/eval_500_10-15_all_${TASK}.jsonl
    $PROJECT_BASE/tactics/data_sets/data/0-11/eval_500_1-5_11_${TASK}.jsonl
    $PROJECT_BASE/tactics/data_sets/data/0-11/eval_500_10-15_11_${TASK}.jsonl
)
EVAL_SIZE=500
TOKENIZER_PATH=$PROJECT_BASE/tactics/data_sets/data/6_tactics_tokenizer

cd ../extrapolate

for EVAL_FILE in ${EVAL_FILES[@]}; do
    python test.py --model_path $MODEL_PATH --eval_file $EVAL_FILE --eval_size $EVAL_SIZE --tokenizer_path $TOKENIZER_PATH
done