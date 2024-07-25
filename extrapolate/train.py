from data_sets.dataloader import JsonlDataset, default_data_collator
from trainer import CustomTrainer
from transformers import TrainingArguments
from models.modeling_mlm import LlamaForMLM, LlamaConfig

DEFAULT_CONFIG=LlamaConfig(
    vocab_size=0,   # tokenizer.vocab_size
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=0, # tokenizer.model_max_length
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,     # tokenizer.pad_token_id
    gradient_checkpointing=False,
    pretraining_tp=1,
    rope_theta=10000,
    rope_scaling=None,
    rms_norm_eps=1e-12,
    attn_implementation="sdpa",
)

DEFAULT_TOKENIZER="./data_sets/data/tokenizer"

import argparse

def parse_args():
    # parser = transformers.HfArgumentParser((TrainingArguments, LlamaConfig))
    parser = argparse.ArgumentParser()

    # add extra arguments
    parser.add_argument("--train_file", type=str, help="Path to the training file")
    parser.add_argument("--eval_file", type=str, help="Path to the evaluation file")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", default="./output")
    return parser.parse_args()


def main():
    args = parse_args()

    train_dataset = JsonlDataset(args.train_file)
    eval_dataset = JsonlDataset(args.eval_file)

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(DEFAULT_TOKENIZER)
    DEFAULT_CONFIG.vocab_size = tokenizer.vocab_size
    DEFAULT_CONFIG.max_position_embeddings = tokenizer.model_max_length
    DEFAULT_CONFIG.pad_token_id = tokenizer.pad_token_id
    print(DEFAULT_CONFIG)

    model = LlamaForMLM(DEFAULT_CONFIG)

    # use cosine with warm up
    training_args = TrainingArguments(
        learning_rate=5e-5,
        output_dir=args.output_dir,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        overwrite_output_dir=True,
        num_train_epochs=300,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=500,
        save_total_limit=5,
        report_to="wandb",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # init wandb
    import wandb
    wandb.init(project='mask-arithmetic', entity='xukp20')

    trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()