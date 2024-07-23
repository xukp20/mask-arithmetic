MODEL_PATH="./def_model_mix_30_wo_20/checkpoint-10"
MODEL_TYPE="l2" # "ce" or "l2"

# Load the model
if MODEL_TYPE == "ce":
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained(MODEL_PATH)
else:
    from modeling_def import LlamaForMLM
    model = LlamaForMLM.from_pretrained(MODEL_PATH)

# load eval data
from train import JSONLDataset
from tokenizer import DEFAULT_TOKENIZER


# view the cos sim between the word embedding
# if MODEL_TYPE == "l2":
#     import torch
#     word_embeddings = model.get_input_embeddings()
#     all_tokens_embedding = word_embeddings.weight # [vocab_size, hidden_size]
#     print(all_tokens_embedding.size())
#     cos_sim = torch.nn.functional.cosine_similarity(all_tokens_embedding.unsqueeze(1), all_tokens_embedding.unsqueeze(0), dim=-1)
#     print(cos_sim)
#     print(cos_sim.size())

#     # see 84
#     cos_sim_84 = cos_sim[84].tolist()
#     # print with 2 decimal places
#     exit(0)

import torch
def l2_model_predict(model, outputs, input_ids, labels):
    # hidden_states in model outputs
    hidden_states = outputs.hidden_states # [bs, seq_len, hidden_size]

    # look for the mask token in the input
    mask_id = DEFAULT_TOKENIZER.get_mask_token_id()
    
    # flatten the hidden states, input_ids and labels
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    input_ids = input_ids.view(-1)
    labels = labels.view(-1)

    mask_indices = (input_ids == mask_id).nonzero(as_tuple=True)
    mask_indices = mask_indices[0]

    # get the hidden states for the mask token
    mask_hidden_states = hidden_states[mask_indices]    # [bs, hidden_size]
    mask_labels = labels[mask_indices]

    # 1. predict, find the smallest l2 between mask hidden and all the tokens
    embedding = model.get_input_embeddings()
    all_tokens_embedding = embedding.weight # [vocab_size, hidden_size]
    mask_labels_embedding = all_tokens_embedding[mask_labels]   # [bs, hidden_size]

    # calculate l2 distance
    # hidden_states: [bs, hidden_size], all_tokens_embedding: [vocab_size, hidden_size] -> [bs, vocab_size]
    l2_distance = torch.cdist(mask_hidden_states, all_tokens_embedding, p=2)   # [bs, vocab_size]
    print(l2_distance)
    # 2. get the top 1
    distance, predicted = torch.topk(l2_distance, 1, largest=False)
    print(distance, predicted)
    predicted = predicted.squeeze(0)

    # Decode
    print(f"Input: {DEFAULT_TOKENIZER.decode(input_ids).strip('[PAD]')}")
    input_ids[mask_indices] = predicted
    print(f"Predicted: {DEFAULT_TOKENIZER.decode(input_ids).strip('[PAD]')}")
    input_ids[mask_indices] = mask_labels
    print(f"Labels: {DEFAULT_TOKENIZER.decode(input_ids).strip('[PAD]')}")
    print()

    # 2. evaluate
    total = mask_labels.size(0)
    correct = (predicted == mask_labels).sum().item()

    # l2 of the predicted and the label
    l2_distance = torch.cdist(mask_hidden_states, mask_labels_embedding, p=2).squeeze(0)   # [bs]

    return correct, total, l2_distance.tolist()


# evaluate
# NOTE: tested on all three eval datasets
# for eval_path in ["./data/eval_ans.jsonl", "./data/eval_num.jsonl", "./data/eval_mix.jsonl"]:
for eval_path in ["./data/eval_eq.jsonl"]:
    eval_dataset = JSONLDataset(eval_path)

    correct = 0
    total = 0
    if MODEL_TYPE == "l2":
        # record the cos sim between the predicted and the label
        l2_distances = []

    for sample in eval_dataset:
        input_ids = sample['input_ids'].unsqueeze(0)
        attention_mask = sample['attention_mask'].unsqueeze(0)
        labels = sample['labels'].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if MODEL_TYPE == "ce":
            logits = outputs.logits
            predicted = logits.argmax(-1)
            
            # look for the mask token in the input
            mask_id = DEFAULT_TOKENIZER.get_mask_token_id()
            mask_idx = (input_ids == mask_id).nonzero(as_tuple=True)
            mask_idx = mask_idx[1]

            # compare the predicted token with the label
            predicted = predicted[0][mask_idx]
            labels = labels[0][mask_idx]

            total += 1
            if predicted == labels:
                correct += 1
        elif MODEL_TYPE == "l2":
            c, t, l2 = l2_model_predict(model, outputs, input_ids, labels)
            correct += c
            total += t
            l2_distances.extend(l2)

    if MODEL_TYPE == "l2":
        print(f"Accuracy for {eval_path}: {correct / total * 100:.2f}%")
        print(f"Average L2 distance: {sum(l2_distances) / len(l2_distances):.2f}")
    else:
        print(f"Accuracy for {eval_path}: {correct / total * 100:.2f}%")