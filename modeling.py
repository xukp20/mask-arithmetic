import torch
from transformers import BertConfig, BertForMaskedLM

def create_custom_bert_model(
    vocab_size=30522,  # Default BERT vocab size
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    pad_token_id=0,
    use_cache=True,
):
    # Create a custom configuration
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=pad_token_id,
        use_cache=use_cache,
    )
    
    # Create the model with random weights
    model = BertForMaskedLM(config)
    
    # Initialize the weights
    model.init_weights()
    
    return model