TRAIN_FILE="./data_sets/data/train_def_0-12_0-11_add&sub_2.jsonl"
EVAL_FILE="./data_sets/data/eval_def_0-12_0-11_add&sub_2.jsonl"

TIMESTAMP=$(date "+%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR="./output/def_model_$TIMESTAMP"

python train.py --train_file $TRAIN_FILE --eval_file $EVAL_FILE --output_dir $OUTPUT_DIR
