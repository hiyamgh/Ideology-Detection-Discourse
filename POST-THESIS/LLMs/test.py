import torch.distributed as dist
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from transformers import EvalPrediction
import torch
from sklearn.model_selection import train_test_split
import time
import argparse
from huggingface_hub import login


def get_response(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    # print(input_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Create attention mask
    attention_mask = attention_mask
    # print(attention_mask)
    inputs = input_ids
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        attention_mask=attention_mask,  # Pass attention mask here
        top_p=0.85,
        temperature=0.9,
        max_length=input_len + 10,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)) + 1:]  # Skip the prompt length

    return response.strip()  # Strip any leading/trailing whitespace


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompting LLMs")
    parser.add_argument("--model_checkpoint", type=str, default="core42/jais-13b", help="The model to use for prediction")
    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint

    # log in to hugging face to be able to access models that need access tokens
    with open("token.txt", "r") as f:
        token = f.read().strip().replace("\n", "")
    login(token=token)

    # Step 1: Initialize the model with empty weights
    model = AutoModelForCausalLM.from_pretrained(
        f"{model_checkpoint}",
        trust_remote_code=True,
        device_map="auto",  # Distributes the model across GPUs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_checkpoint}",
        trust_remote_code=True
    )

    # evaluate the model for 10 prompts
    prompts = [
        "My favourite condiment is",
        "My favourite condiment is",
    ]
    for prompt in prompts:
        answer = get_response(text=prompt)
        print(answer)