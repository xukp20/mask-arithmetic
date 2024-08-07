# change dir to ../extrapolate to PYTHONPATH
cd ../extrapolate
# train the model

PROJECT_PATH=/cephfs/xukangping/code/mask-arithmetic
DATA_BASE=$PROJECT_PATH/tactics/data_sets/data

VALUES=0-11
LENGTH_RANGE=1-5
TRAIN_SIZE=4000
EVAL_SIZE=500
TASK=position
CONTAIN_NUM=11

TRAIN_FILES=(
    $DATA_BASE/$VALUES/train_${TRAIN_SIZE}_${LENGTH_RANGE}_${CONTAIN_NUM}_${TASK}.jsonl
)

TRAIN_SIZES=(
    $TRAIN_SIZE
)

EVAL_FILES=(
    $DATA_BASE/$VALUES/eval_${EVAL_SIZE}_${LENGTH_RANGE}_${CONTAIN_NUM}_${TASK}.jsonl
)

EVAL_SIZES=(
    $EVAL_SIZE
)

ONE_HOT=0
# TRAIN_ONLY_EMBEDDINGS=11

OUTPUT_DIR=$PROJECT_PATH/tactics/outputs/${TASK}_${VALUES}_${LENGTH_RANGE}
if [ $ONE_HOT -eq 1 ]; then
    OUTPUT_DIR=$OUTPUT_DIR-one-hot
fi

TOKENIZER_PATH=$PROJECT_PATH/tactics/data_sets/data/6_tactics_tokenizer

ALPHA=0

LEARNING_RATE=1e-2
# LEARNING_RATE=1e-4
EPOCHS=100

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

if [ $ONE_HOT -eq 1 ]; then
    command+=" --one_hot_embedding --fix_embedding"
fi

TRAIN_ONLY_EMBEDDINGS=11

if [ ! -z $TRAIN_ONLY_EMBEDDINGS ]; then
    for ID in ${TRAIN_ONLY_EMBEDDINGS[@]}; do
        command+=" --train_only_embeddings $ID"
    done
fi

LOAD_MODEL=$PROJECT_PATH/tactics/outputs/${TASK}_0-10_10-15

if [ ! -z $LOAD_MODEL ]; then
    command+=" --load_model $LOAD_MODEL"
fi

echo $command

# export wandb run name
RUN_NAME=${TASK}_${VALUES}_${LENGTH_RANGE}_${TRAIN_SIZE}_${EVAL_SIZE}

export WANDB_NAME=$RUN_NAME

$command