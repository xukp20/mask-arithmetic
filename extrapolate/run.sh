# TRAIN_FILE="./data_sets/data/train_def_0-12_0-11_add&sub_2.jsonl"
# EVAL_FILE="./data_sets/data/eval_def_0-12_0-11_add&sub_2.jsonl"

TRAIN_FILES=(
    /cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/train_equal_0-11_add\&sub_2_2_2_-1_any.jsonl
    /cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/train_def_0-12_0-11_add\&sub_2.jsonl
)

EVAL_FILES=(
    /cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/eval_equal_0-11_add\&sub_2_2_2_-1_any.jsonl
    /cephfs/xukangping/code/mask-arithmetic/extrapolate/data_sets/data/eval_def_0-12_0-11_add\&sub_2.jsonl
)

TRAIN_SIZES=(
    2000
    2000
)

EVAL_SIZES=(
    50
    50
)


TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR="./output/def_model_$TIMESTAMP"

command="python train.py --output_dir $OUTPUT_DIR"

for TRAIN_FILE in "${TRAIN_FILES[@]}"; do
    command="$command --train_files $TRAIN_FILE"
done

for EVAL_FILE in "${EVAL_FILES[@]}"; do
    command="$command --eval_files $EVAL_FILE"
done

for TRAIN_SIZE in "${TRAIN_SIZES[@]}"; do
    command="$command --train_sizes $TRAIN_SIZE"
done

for EVAL_SIZE in "${EVAL_SIZES[@]}"; do
    command="$command --eval_sizes $EVAL_SIZE"
done

echo $command
$command

