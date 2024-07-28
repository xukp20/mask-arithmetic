# change dir to ../extrapolate to PYTHONPATH
cd ../extrapolate
# train the model

PROJECT_PATH=/cephfs/xukangping/code/mask-arithmetic

TRAIN_FILES=(
    # $PROJECT_PATH/tactics/data_sets/data/train_baseline_tactic.jsonl
    # $PROJECT_PATH/tactics/data_sets/data/train_baseline_both.jsonl
    $PROJECT_PATH/tactics/data_sets/data/train_baseline_position.jsonl
)

TRAIN_SIZES=(
    # 100000
    100000
)

EVAL_FILES=(
    # $PROJECT_PATH/tactics/data_sets/data/eval_baseline_tactic.jsonl
    # $PROJECT_PATH/tactics/data_sets/data/eval_baseline_both.jsonl
    $PROJECT_PATH/tactics/data_sets/data/eval_baseline_position.jsonl
)

EVAL_SIZES=(
    500
    # 500
)

OUTPUT_DIR=$PROJECT_PATH/tactics/outputs/position
TOKENIZER_PATH=$PROJECT_PATH/tactics/data_sets/data/6_tactics_tokenizer

ALPHA=0

LEARNING_RATE=5e-5
EPOCHS=10

command="python train.py \
    --output_dir $OUTPUT_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --alpha $ALPHA \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS"

for i in ${!TRAIN_FILES[@]}; do
    command+=" --train_files ${TRAIN_FILES[$i]} --train_sizes ${TRAIN_SIZES[$i]}"
done

for i in ${!EVAL_FILES[@]}; do
    command+=" --eval_files ${EVAL_FILES[$i]} --eval_sizes ${EVAL_SIZES[$i]}"
done

echo $command

$command