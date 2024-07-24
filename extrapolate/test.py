from data_sets.dataloader import JsonlDataset, create_dataloader
from models.modeling_mlm import LlamaForMLM, LlamaConfig
import torch

DEFAULT_TOKENIZER="./data_sets/data/tokenizer"

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, help="Path to the evaluation file")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="./output")

    return parser.parse_args()


def main():
    args = parse_args()

    dataloader = create_dataloader(args.eval_file, batch_size=1)

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(DEFAULT_TOKENIZER)

    model = LlamaForMLM.from_pretrained(args.model_path)
    model.eval()

    def l2_predict(model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # look for mask token in the input
        mask_id = tokenizer.convert_tokens_to_ids("<mask>")

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
        all_tokens_embedding = embedding.weight
        mask_labels_embedding = all_tokens_embedding[mask_labels]

        # calculate l2 distance
        l2_distance = torch.cdist(mask_hidden_states, all_tokens_embedding, p=2)
        # 2. get top 5
        top5_l2 = torch.topk(l2_distance, 5, largest=False, sorted=True)[0]
        # 3. predicted is the top 1
        predicted = torch.argmax(l2_distance, dim=-1)
        correct = (predicted == mask_labels).sum().item()
        total = mask_labels.size(0)
        # 4. compute the distance of label and output
        l2_distance = torch.cdist(mask_hidden_states, mask_labels_embedding, p=2).squeeze(0)

        # print(l2_distance.shape)
        # print(top5_l2.shape)
        return correct, total, l2_distance, top5_l2

    def mlm_predict(model, input_ids, attention_mask, labels):
        pass
        # TODO
        
    correct = 0
    total = 0
    l2_distance = []
    top5_l2 = {i: [] for i in range(5)}
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        correct_, total_, l2_distance_, top5_l2_ = predict(model, input_ids, attention_mask, labels)
        correct += correct_
        total += total_
        l2_distance.extend(l2_distance_)
        for i in range(5):
            top5_l2[i].extend(top5_l2_[:, i].tolist())
        
    print(f"Accuracy: {correct / total}")
    print(f"Average L2 distance between label and output presheaf: {sum(l2_distance) / len(l2_distance)}")
    for i in range(5):
        print(f"Top {i + 1} average L2 distance: {sum(top5_l2[i]) / len(top5_l2[i])}")


if __name__ == "__main__":
    main()