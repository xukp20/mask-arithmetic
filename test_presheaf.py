"""
    Check if the def model can generate a presheaf that makes sense.
"""

from tokenizer import DEFAULT_TOKENIZER, MAX_NUMBER
from generate_data import MOD
def load_model(model_path, device):
    from modeling_def import LlamaForMLM
    model = LlamaForMLM.from_pretrained(model_path)
    model.to(device)

    return model


import re
import torch

def interactive_generation(model, tokenizer, device):
    # take out the embed tokens of the model
    word_embeddings = model.get_input_embeddings()

    # keep track of the def presheaf
    def_presheaf = {}
    print(f"""Interactive generation:
Configs: max_number: {MAX_NUMBER}, mod: {MOD}
""")
    while True:
        command = input("Enter command:\n - 'e' to input a expr for generating a presheaf\n - 'd' to decode a saved presheaf\n - 'q' to quit\n>>> ")
        if command == 'q':
            break
        elif command == 'e':
            expr = input("Enter an expression (Use [MASK] for the place to generate the presheaf, [DEF(i)] for insert the i-th saved presheaf):\n")
            # first handle the [DEF(i)] 
            split_expr = expr.split()
            # tokenized the expr
            tokenized_expr = tokenizer.encode(expr)
            # get the mask token
            mask_id = tokenizer.get_mask_token_id()
            mask_positions = torch.tensor([i == mask_id for i in tokenized_expr['input_ids']], dtype=torch.bool)
            # print("Mask:", mask_positions)

            # look for def tokens
            success = True
            presheaf_tensors = []
            for i, token in enumerate(split_expr):
                if token.startswith("[DEF(") and token.endswith(")]"):
                    def_id = token[5:-2]
                    if def_id not in def_presheaf:
                        print(f"Def {def_id} not found")
                        success = False
                        break
                    else:
                        presheaf_tensors.append(def_presheaf[def_id])

            if not success:
                continue
        
            # generate the input_embeds
            # check unk count == def count
            unk_id = tokenizer.unk_token_id
            unk_positions = torch.tensor([i == unk_id for i in tokenized_expr['input_ids']], dtype=torch.bool)
            unk_count = unk_positions.sum().item()
            unk_positions = torch.where(unk_positions)[0]
            def_count = len(presheaf_tensors)
            if unk_count != def_count:
                print(f"Unk count {unk_count} != def count {def_count}, there is unknown tokens in the expression")
                continue

            # get the input_embeds
            inputs_embeds = word_embeddings(torch.tensor(tokenized_expr['input_ids']).to(device))
            # print("Input embeds:", inputs_embeds.size())
            # replace the unk with the def
            for i, def_tensor in enumerate(presheaf_tensors):
                inputs_embeds[unk_positions[i]] = def_tensor
                # print("Replace:", unk_positions[i], split_expr[unk_positions[i]])
                # print(inputs_embeds[unk_positions[i]].size())
                # print(def_tensor.size())
            
            # create a batch to forward
            attention_mask = torch.tensor(tokenized_expr['attention_mask']).unsqueeze(0).to(device)
            inputs_embeds = inputs_embeds.unsqueeze(0)
            inputs_embeds = inputs_embeds.to(device)
            mask_positions = mask_positions.unsqueeze(0).to(device)

            # get mask hidden states
            hidden_states = model.get_mask_hidden_states(inputs_embeds=inputs_embeds, attention_mask=attention_mask, mask_positions=mask_positions)

            # mask count
            mask_count = mask_positions.sum().item()
            hidden_length = hidden_states.size(0)
            if mask_count != hidden_length:
                print(f"Mask count {mask_count} != hidden length {hidden_length}")
                continue
        
            # can save
            print(expr)
            for i, mask_presheaf in enumerate(hidden_states):
                save = input(f"Save the {i}-th mask presheaf? (y/n)")
                if save == 'y':
                    key = input("Enter the key:")
                    def_presheaf[key] = mask_presheaf

                decode = input(f"Decode the {i}-th mask presheaf? (y/n)")
                if decode == 'y':
                    # get the l2 distance
                    all_tokens_embedding = word_embeddings.weight
                    l2_distance = torch.cdist(mask_presheaf.unsqueeze(0), all_tokens_embedding, p=2)
                    distance, predicted = torch.topk(l2_distance, 5, largest=False)
                    tokens = [tokenizer.decode(i.item()) for i in predicted.squeeze(0)]
                    for i, token in enumerate(tokens):
                        print(f"{i+1}: {token}, distance: {distance[0, i].item()}")

            print("Updated keys:", def_presheaf.keys())

        elif command == 'd':
            # look for the token with the smallest l2 distance 
            print(def_presheaf.keys())
            key = input("Enter the key to decode:")
            if key not in def_presheaf:
                print(f"Key {key} not found")
                continue
            else:
                presheaf = def_presheaf[key]
                # get the l2 distance
                all_tokens_embedding = word_embeddings.weight   # [vocab_size, hidden_size]
                l2_distance = torch.cdist(presheaf.unsqueeze(0), all_tokens_embedding, p=2)  # [1, vocab_size]
                distance, predicted = torch.topk(l2_distance, 5, largest=False)
                # print the top 5
                tokens = [tokenizer.decode(i.item()) for i in predicted.squeeze(0)]
                for i, token in enumerate(tokens):
                    print(f"{i+1}: {token}, distance: {distance[0, i].item()}")
        else:
            print("Unknown command")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./def_model_mix_10")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model = load_model(args.model_path, args.device)
    interactive_generation(model, DEFAULT_TOKENIZER, args.device)
                