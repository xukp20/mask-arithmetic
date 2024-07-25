"""
    Check if the def model can generate a presheaf that makes sense.
"""

from train import DEFAULT_TOKENIZER
def load_model(model_path, device):
    from models.modeling_mlm import LlamaForMLM
    model = LlamaForMLM.from_pretrained(model_path)
    model.to(device)

    return model


import re
import torch

def interactive_generation(model, tokenizer_path, device):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # take out the embed tokens of the model
    word_embeddings = model.get_input_embeddings()

    # keep track of the def presheaf
    def_presheaf = {}
    print(f"""Interactive generation:
""")
    while True:
        command = input("""
Enter command:
 - 'e' to input a expr for generating a presheaf
 - 'd' to decode a saved presheaf
 - 'm' to measure the distance between two presheafs
 - 'q' to quit
 >>> """)
        if command == 'q':
            break
        elif command == 'e':
            expr = input("Enter an expression (Use <mask> for the place to generate the presheaf, <def(i)> for insert the i-th saved presheaf):\n")
            # first handle the def
            split_expr = expr.split()
            # tokenized the expr
            tokenized_expr = [tokenizer.convert_tokens_to_ids(token) for token in split_expr]
            # get the mask token
            mask_id = tokenizer.convert_tokens_to_ids("<mask>")
            mask_positions = torch.tensor([i == mask_id for i in tokenized_expr], dtype=torch.bool)

            # look for def tokens
            success = True
            presheaf_tensors = []
            new_tokens = []
            for i, token in enumerate(split_expr):
                if token.startswith("<def(") and token.endswith(")>"):
                    def_id = token[5:-2]
                    if def_id not in def_presheaf:
                        print(f"Def {def_id} not found")
                        success = False
                        break
                    else:
                        presheaf_tensors.append(def_presheaf[def_id])
                    new_tokens.append("<unk>")
                else:
                    new_tokens.append(token)

            if not success:
                continue
        
            # generate the input_embeds
            # check unk count == def count
            unk_id = tokenizer.unk_token_id
            unk_positions = torch.tensor([i == unk_id for i in tokenized_expr], dtype=torch.bool)
            unk_count = unk_positions.sum().item()
            unk_positions = torch.where(unk_positions)[0]
            def_count = len(presheaf_tensors)
            if unk_count != def_count:
                print(f"Unk count {unk_count} != def count {def_count}, there is unknown tokens in the expression")
                print("Unk positions:", unk_positions)
                print(tokenized_expr)
                print(tokenizer.decode(tokenized_expr))
                continue

            # get the input_embeds
            inputs_embeds = word_embeddings(torch.tensor(tokenized_expr).to(device))
            # replace the unk with the def
            for i, def_tensor in enumerate(presheaf_tensors):
                inputs_embeds[unk_positions[i]] = def_tensor
            
            # create a batch to forward
            expr = " ".join(new_tokens)
            attention_mask = tokenizer(expr, return_tensors="pt")["attention_mask"].to(device)
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
        
        elif command == 'm':
            print(def_presheaf.keys())
            key1 = input("Enter the key of the first presheaf:")
            key2 = input("Enter the key of the second presheaf:")
            if key1 not in def_presheaf:
                print(f"Key {key1} not found")
                continue
            if key2 not in def_presheaf:
                print(f"Key {key2} not found")
                continue

            presheaf1 = def_presheaf[key1]
            presheaf2 = def_presheaf[key2]
            l2_distance = torch.cdist(presheaf1.unsqueeze(0), presheaf2.unsqueeze(0), p=2)
            print(f"L2 distance between {key1} and {key2}: {l2_distance.item()}")

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
                