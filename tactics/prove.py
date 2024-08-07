"""
    Given a real source and target expression, use the tactic model and position model (could be the same one)
"""

PROJECT_BASE="../"
TACTIC_DATASET_BASE="./data_sets"
import sys
sys.path.append(PROJECT_BASE)
sys.path.append(TACTIC_DATASET_BASE)

from extrapolate.models.modeling_mlm import LlamaForMLM

import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Prove a given source and target expression')
    parser.add_argument('--tactic_model_path', type=str, help='Path to the tactic model', required=True)
    parser.add_argument('--position_model_path', type=str, default=None, help='Path to the position model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer', required=True)
    parser.add_argument('--test_file', type=str, help='Path to the test file', default=None)
    parser.add_argument('--test_size', type=int, help='Number of test cases to generate', default=100)
    parser.add_argument('--random_seed', type=int, help='Random seed', default=42)
    parser.add_argument('--device', type=str, help='Device to use', default='cuda')
    parser.add_argument('--log_dir', type=str, help='Path to the log directory', default=None)
    
    # data setting
    parser.add_argument('--contain_number', type=int, help='Must contained number in the expression', default=None)

    # prove settings
    parser.add_argument('--max_steps', type=int, help='Maximum number of steps to take', default=30)
    parser.add_argument('--lower', type=int, help='Lower bound of numbers', default=0)
    parser.add_argument('--upper', type=int, help='Upper bound of numbers', default=10)
    return parser.parse_args()


def load_model(args):
    tactic_model = LlamaForMLM.from_pretrained(args.tactic_model_path)
    tactic_model.to(args.device)
    tactic_model.eval()
    if args.position_model_path is not None:
        position_model = LlamaForMLM.from_pretrained(args.position_model_path)
        position_model.to(args.device)
        position_model.eval()
    else:
        position_model = tactic_model
    
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    return tactic_model, position_model, tokenizer


import json
from data_sets.expr import Expr
from data_sets.proof import Proof, ProofStep
from data_sets.tactic import Tactic, ALL_TACTICS
from data_sets.generate_tactic_datasets import (
    TACTIC_LENGTH, POSITION_LENGTH, EXPR_LENGTH, MAX_LENGTH, MASK_TOKEN, get_attn_mask, get_sample_ids, get_labels,
    TACTIC_TO_TOKEN, TOKEN_TO_TACTIC
)

def mlm_inference(model, input_ids, attention_mask, mask_id):
    unsqueeze = False
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        unsqueeze = True
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.hidden_states

    # look for mask token in the input
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    input_ids = input_ids.view(-1)
    
    mask_indices = (input_ids == mask_id).nonzero(as_tuple=True)
    mask_indices = mask_indices[0]

    # get the hidden states for the mask token
    mask_hidden_states = hidden_states[mask_indices]    # [bs, hidden_size]

    # compute the logits
    embedding_weights = model.get_input_embeddings().weight
    logits = torch.matmul(mask_hidden_states, embedding_weights.T)

    # take top1
    top1 = torch.argmax(logits, dim=-1)

    return top1
    
# get filters
def get_proof_filter(args):
    # the contain number appears in one of the exprs 
    if not args.contain_number:
        return lambda proof: True
    else:
        def proof_filter(proof):
            if args.contain_number in proof.source.nums or args.contain_number in proof.target.nums:
                return True
            for step in proof.steps:
                if args.contain_number in step.expr.nums:
                    return True
            return False

        return proof_filter

def batch_prove(samples, tactic_model, position_model, tokenizer, args):
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)

    status = {
        "success": 0,
        "failed": 0,
        "length_exceeded": 0,
        "better_than_reference": 0,
        "worse_than_reference": 0,
        "same_as_reference": 0,
        "error": 0,
    }

    for source, target, length in samples:
        try:
            current_length = 0
            success = False
            start = source
            while current_length < args.max_steps:
                # get inputs of tactic task
                tactic_sample = {
                    'source': str(source),
                    'target': str(target),
                    'tactic': MASK_TOKEN,
                    'position': MASK_TOKEN,
                }
                input_ids = get_sample_ids(tactic_sample, tokenizer, mask_id=mask_id, task="tactic")
                input_ids = torch.tensor(input_ids).to(args.device)
                attention_mask = get_attn_mask(tactic_sample, tokenizer)
                attention_mask = torch.tensor(attention_mask).to(args.device)

                # get the tactic
                tactic_id = mlm_inference(tactic_model, input_ids, attention_mask, mask_id)
                tactic = tokenizer.convert_ids_to_tokens(tactic_id)
                assert len(tactic) == 1 and tactic[0] in TOKEN_TO_TACTIC
                tactic = tactic[0]

                # get inputs of position task
                position_sample = {
                    'source': str(source),
                    'target': str(target),
                    'tactic': tactic,
                    'position': MASK_TOKEN,
                }
                input_ids = get_sample_ids(position_sample, tokenizer, mask_id=mask_id, task="position")
                input_ids = torch.tensor(input_ids).to(args.device)
                attention_mask = get_attn_mask(position_sample, tokenizer, task="position")
                attention_mask = torch.tensor(attention_mask).to(args.device)
                # get the position
                position_id = mlm_inference(position_model, input_ids, attention_mask, mask_id)
                position = tokenizer.convert_ids_to_tokens(position_id)
                assert len(position) == 1 and position[0].isdigit()
                position = position[0]

                # get the proof step
                tactic_name = TOKEN_TO_TACTIC[tactic]
                position = int(position)
                tactic = ALL_TACTICS[tactic_name]
                # apply to source
                expr = tactic.apply(source, position, args.lower, args.upper)

                if not expr:
                    print(f"Failed to prove: {source} -> {target}")
                    print(f"Tactic: {tactic_name}, Position: {position} cannot be applied")
                    status["failed"] += 1
                    break
                else:
                    print(f"    Applied tactic {tactic_name} at position {position}: {source} -> {expr}")
                    source = expr
                    current_length += 1
                    if source == target:
                        success = True
                        status["success"] += 1
                        break
            if success:
                # compare the lengths
                print(f"Proved: {start} -> {target}, use {current_length} steps, reference length: {length}")
                if current_length < length:
                    status["better_than_reference"] += 1
                elif current_length == length:
                    status["same_as_reference"] += 1
                else:
                    status["worse_than_reference"] += 1
            elif current_length >= length:
                print(f"Length exceeded: {start} -> {target}. Reference length: {length}")
                status["length_exceeded"] += 1
        except Exception as e:
            print(f"Error: {e}")
            status["error"] += 1

    # log
    print(f"Success: {status['success']}")
    print(f"Failed: {status['failed']}")
    print(f"Length exceeded: {status['length_exceeded']}")
    print(f"Better than reference: {status['better_than_reference']}")
    print(f"Worse than reference: {status['worse_than_reference']}")
    print(f"Same as reference: {status['same_as_reference']}")
    print(f"Error: {status['error']}")
    status["acc"] = status["success"] / len(samples)
    print(f"Accuracy: {status['acc']}")
    status["error_rate"] = status["error"] / len(samples)
    print(f"Error rate: {status['error_rate']}")
    status["failed_rate"] = status["failed"] / len(samples)
    print(f"Failed rate: {status['failed_rate']}")

    return status


def single_solve(source, target, tactic_model, position_model, tokenizer, args):
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)

    current_length = 0
    success = False
    while current_length < args.max_steps:
        # get inputs of tactic task
        tactic_sample = {
            'source': source,
            'target': target,
            'tactic': MASK_TOKEN,
            'position': MASK_TOKEN,
        }
        input_ids = get_sample_ids(tactic_sample, tokenizer, mask_id=mask_id, task="tactic")
        input_ids = torch.tensor(input_ids).to(args.device)
        attention_mask = get_attn_mask(tactic_sample, tokenizer)
        attention_mask = torch.tensor(attention_mask).to(args.device)

        # get the tactic
        tactic_id = mlm_inference(tactic_model, input_ids, attention_mask, mask_id)
        tactic = tokenizer.convert_ids_to_tokens(tactic_id)

        # get inputs of position task
        position_sample = {
            'source': source,
            'target': target,
            'tactic': tactic,
            'position': MASK_TOKEN,
        }
        input_ids = get_sample_ids(position_sample, tokenizer, mask_id=mask_id, task="position")
        input_ids = torch.tensor(input_ids).to(args.device)
        attention_mask = get_attn_mask(position_sample, tokenizer)
        attention_mask = torch.tensor(attention_mask).to(args.device)

        # get the position
        position_id = mlm_inference(position_model, input_ids, attention_mask, mask_id)
        position = tokenizer.convert_ids_to_tokens(position_id)

        # get the proof step
        tactic_name = TOKEN_TO_TACTIC[tactic]
        position = int(position)
        tactic = ALL_TACTICS[tactic_name]
        # apply to source
        expr = tactic.apply(source, position, args.lower, args.upper)

        if not expr.equals(target):
            print(f"Failed to prove: {source} -> {target}")
            print(f"Tactic: {tactic_name}, Position: {position} cannot be applied")
            break
        else:
            print(f"Applied tactic {tactic_name} at position {position}: {source} -> {expr}")
            source = expr
            current_length += 1
            if source.equals(target):
                success = True
                break
    if success:
        print(f"Proved: {source} -> {target}, use {current_length} steps")
    else:
        print(f"Failed to prove: {source} -> {target}")


import random
def prove(args):
    if args.test_file is not None:
        with open(args.test_file, 'r') as f:
            tests = [json.loads(line) for line in f]
        
        samples = []
        print("There are", len(tests), "proofs in total")
        # tests = tests[:args.test_size]
        # filter the proofs
        proof_filter = get_proof_filter(args)
        bitmap = [False] * len(tests)
        for i in range(len(tests)):
            if proof_filter(Proof.from_dict(tests[i])):
                bitmap[i] = True
        tests = [test for i, test in enumerate(tests) if bitmap[i]]
        print("After filtering, there are", len(tests), "proofs left")
        random.seed(args.random_seed)
        tests = random.sample(tests, min(args.test_size, len(tests)))

        for test in tests:
            proof = Proof.from_dict(test)
            source = proof.source
            target = proof.target
            length = len(proof.steps)
            samples.append((source, target, length))
        
        status = batch_prove(samples, tactic_model, position_model, tokenizer, args)
        return status
    else:
        while True:
            source = input("Enter the source expression: ")
            target = input("Enter the target expression: ")
            source = Expr.from_string(source)
            target = Expr.from_string(target)
            single_solve(source, target, tactic_model, position_model, tokenizer, args)

    
import os
if __name__ == "__main__":
    args = parse_args()
    tactic_model, position_model, tokenizer = load_model(args)
    status = prove(args)    
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        basename = os.path.basename(args.test_file).strip(".jsonl")
        with open(os.path.join(args.log_dir, f"{basename}.json"), 'w') as f:
            json.dump(status, f, indent=2)
