PROJECT_BASE=/cephfs/xukangping/code/mask-arithmetic
TACTIC_MODEL_PATH=$PROJECT_BASE/tactics/outputs/tactic_0-10_10-15
POSITION_MODEL_PATH=$PROJECT_BASE/tactics/outputs/position_0-10_10-15
TEST_FILE=$PROJECT_BASE/tactics/data_sets/data/0-11/eval_500_10-15_11_tactic.jsonl
TEST_SIZE=100
CONTAIN_NUMBER=11

LOG_DIR=$PROJECT_BASE/tactics/outputs/prove
TOKENIZER_PATH=$PROJECT_BASE/tactics/data_sets/data/6_tactics_tokenizer

if [ ! -z $POSITION_MODEL_PATH ]; then
    POSITION_MODEL_PATH="--position_model_path $POSITION_MODEL_PATH"
else
    POSITION_MODEL_PATH=""
fi

# from 1 to 25
for LENGTH in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do

    TEST_FILE=$PROJECT_BASE/tactics/data_sets/data/0-11/proof_num_op2-5/$LENGTH.jsonl
    FILE_STR="--test_file $TEST_FILE --test_size $TEST_SIZE"

    command="python prove.py \
        --tactic_model_path $TACTIC_MODEL_PATH \
        $FILE_STR \
        $POSITION_MODEL_PATH \
        --tokenizer_path $TOKENIZER_PATH \
        --log_dir $LOG_DIR \
        --contain_number $CONTAIN_NUMBER"

    echo $command

    $command
done