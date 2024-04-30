import argparse
import sys
import csv
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from transformers import TextStreamer
import pandas as pd

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200):
    prompt_chunks = prompt.split("<image>")
    tokenized_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
    input_ids = tokenized_chunks[0]

    for chunk in tokenized_chunks[1:]:
        input_ids.append(image_token_index)
        input_ids.extend(chunk[1:])  # Exclude BOS token on nonzero index

    return torch.tensor(input_ids, dtype=torch.long)


def process_tensors(input_ids, image_features, embedding_layer):
    # Find the index of -200 in input_ids
    split_index = (input_ids == -200).nonzero(as_tuple=True)[1][0]

    # Split the input_ids at the index found, excluding -200
    input_ids_1 = input_ids[:, :split_index]
    input_ids_2 = input_ids[:, split_index + 1 :]

    # Convert input_ids to embeddings
    embeddings_1 = embedding_layer(input_ids_1)
    embeddings_2 = embedding_layer(input_ids_2)

    device = image_features.device
    token_embeddings_part1 = embeddings_1.to(device)
    token_embeddings_part2 = embeddings_2.to(device)

    # Concatenate the token embeddings and image features
    concatenated_embeddings = torch.cat(
        [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
    )

    # Create the corrected attention mask
    attention_mask = torch.ones(
        concatenated_embeddings.shape[:2], dtype=torch.long, device=device
    )
    return concatenated_embeddings, attention_mask


def initialize_models():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/llama-3-8b-Instruct", use_fast=True
    )
    model = LlamaForCausalLM.from_pretrained(
        "unsloth/llama-3-8b-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    for param in model.base_model.parameters():
        param.requires_grad = False

    model_name = "google/siglip-so400m-patch14-384"
    vision_model = SiglipVisionModel.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    processor = SiglipImageProcessor.from_pretrained(model_name)

    vision_model = vision_model.to("cuda")

    return tokenizer, model, vision_model, processor


class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(ProjectionModule, self).__init__()

        # Directly set up the sequential model
        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)


def load_projection_module(mm_hidden_size=1152, hidden_size=4096, device="cuda"):
    projection_module = ProjectionModule(mm_hidden_size, hidden_size)
    checkpoint = torch.load("./checkpoints/llama-3/checkpoint-2300/mm_projector.bin")
    checkpoint = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}
    projection_module.load_state_dict(checkpoint)
    projection_module = projection_module.to(device).half()
    return projection_module

def evaluate_model_and_save_csv(dataset, tokenizer, model, vision_model, processor, projection_module, output_file="evaluation_results.csv"):
    results = []
    for example in dataset:
        image = example['image']
        if isinstance(image, str):  # If image is a path
            image = Image.open(image).convert("RGB")
        # Assuming image is already an Image object if not a path
        elif not isinstance(image, Image.Image):
            continue  # Skip if it's neither a path nor an Image object

        question = example['question']
        correct_answer = example['answer']

        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = (
            tokenizer_image_token(prompt, tokenizer)
            .unsqueeze(0)
            .to(model.device)
        )
        with torch.inference_mode():
            image_inputs = processor(images=[image], return_tensors="pt", do_resize=True, size={"height": 384, "width": 384}).to("cuda")
            image_features = vision_model(image_inputs["pixel_values"].squeeze(0).unsqueeze(0), output_hidden_states=True).hidden_states[-2]
            projected_embeddings = projection_module(image_features).to("cuda")

            embedding_layer = model.get_input_embeddings()
            new_embeds, attn_mask = process_tensors(input_ids, projected_embeddings, embedding_layer)

            model_kwargs = {
                "do_sample": False,  # For evaluation, deterministic output can be better
                "temperature": 0.2,
                "max_new_tokens": 2000,
                "use_cache": True
            }

            generated_ids = model.generate(inputs_embeds=new_embeds, attention_mask=attn_mask, **model_kwargs)[0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            'Image Path': image if isinstance(image, str) else 'Image loaded directly',
            'Question': question,
            'Correct Answer': correct_answer,
            'Generated Answer': generated_text
        })

    # Create a DataFrame and write to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    dataset = load_dataset("xai-org/RealworldQA", split='test').select(range(200))  # Selecting first 200 rows
    tokenizer, model, vision_model, processor = initialize_models()
    projection_module = load_projection_module()

    evaluate_model_and_save_csv(dataset, tokenizer, model, vision_model, processor, projection_module)
