from data_sets.dataloader import JsonlDataset, create_dataloader
from models.modeling_mlm import LlamaForMLM, LlamaConfig
import torch

DEFAULT_TOKENIZER="./data_sets/data/tokenizer"

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, help="Path to the evaluation file")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="./output")
    parser.add_argument("--eval_size", type=int, help="The size of the evaluation file", default=None)
    return parser.parse_args()


from tqdm import tqdm
def main():
    args = parse_args()

    dataloader = create_dataloader(args.eval_file, batch_size=1, sizes=args.eval_size)

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(DEFAULT_TOKENIZER)

    model = LlamaForMLM.from_pretrained(args.model_path, device_map="cuda:0" if torch.cuda.is_available() else "cpu")
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
        # l2_distance = torch.cdist(mask_hidden_states, all_tokens_embedding, p=2)
        # try compute by hand
        l2_distance = torch.sum((mask_hidden_states.unsqueeze(1) - all_tokens_embedding.unsqueeze(0)) ** 2, dim=-1).sqrt()
        # 2. get top 5
        top5_l2 = torch.topk(l2_distance, 5, largest=False, sorted=True)[0]
        # 3. predicted is the top 1
        predicted = torch.argmin(l2_distance, dim=-1)
        correct = (predicted == mask_labels).sum().item()
        total = mask_labels.size(0)
        # 4. compute the distance of label and output
        l2_distance = torch.cdist(mask_hidden_states, mask_labels_embedding, p=2).squeeze(0)

        # print(l2_distance.shape)
        # print(top5_l2.shape)
        return correct, total, l2_distance, top5_l2

    def mlm_predict(model, input_ids, attention_mask, labels):
        # do dot-product between hidden states and the embedding weights to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # look for mask token in the input
        mask_id = tokenizer.convert_tokens_to_ids("<mask>")
        
        # flatten 
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        input_ids = input_ids.view(-1)
        labels = labels.view(-1)

        mask_indices = (input_ids == mask_id).nonzero(as_tuple=True)
        mask_indices = mask_indices[0]

        # get the hidden states for the mask token
        mask_hidden_states = hidden_states[mask_indices]    # [mask_count, hidden_size]
        mask_labels = labels[mask_indices]

        # compute logits
        embedding_weights = model.get_input_embeddings().weight # [vocab_size, hidden_size]
        logits = torch.matmul(mask_hidden_states, embedding_weights.T) # [mask_count, vocab_size]

        # get top 5
        top5_logits = torch.topk(logits, 5, largest=True, sorted=True)[0]
        # get the predicted
        predicted = torch.argmax(logits, dim=-1)
        correct = (predicted == mask_labels).sum().item()
        total = mask_labels.size(0)

        # logits for the correct label
        correct_logits = logits[torch.arange(mask_labels.size(0)), mask_labels]

        return correct, total, correct_logits, top5_logits
        
    correct = 0
    total = 0
    l2_distance = []
    top5_l2 = {i: [] for i in range(5)}
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        correct_, total_, l2_distance_, top5_l2_ = l2_predict(model, input_ids, attention_mask, labels)
        correct += correct_
        total += total_
        l2_distance.extend(l2_distance_)
        for i in range(5):
            top5_l2[i].extend(top5_l2_[:, i].tolist())
        
    print(f"Accuracy: {correct / total}")
    print(f"Average L2 distance between label and output presheaf: {sum(l2_distance) / len(l2_distance)}")
    for i in range(5):
        print(f"Top {i + 1} average L2 distance: {sum(top5_l2[i]) / len(top5_l2[i])}")

    # eval again with mlm_predict
    dataloader = create_dataloader(args.eval_file, batch_size=1, sizes=args.eval_size)
    correct = 0
    total = 0
    correct_logits = []
    top5_logits = {i: [] for i in range(5)}
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        correct_, total_, correct_logits_, top5_logits_ = mlm_predict(model, input_ids, attention_mask, labels)
        correct += correct_
        total += total_
        correct_logits.extend(correct_logits_)
        for i in range(5):
            top5_logits[i].extend(top5_logits_[:, i].tolist())
        
    print(f"Accuracy: {correct / total}")
    print(f"Average logits distance between label and output presheaf: {sum(correct_logits) / len(correct_logits)}")
    for i in range(5):
        print(f"Top {i + 1} average logits distance: {sum(top5_logits[i]) / len(top5_logits[i])}")


if __name__ == "__main__":
    main()