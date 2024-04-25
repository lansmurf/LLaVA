import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM, SiglipImageProcessor, SiglipVisionModel

)



def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return torch.tensor(input_ids, dtype=torch.long)


def process_tensors(input_ids, image_features, embedding_layer):
    # Find the index of -200 in input_ids
    split_index = (input_ids == -200).nonzero(as_tuple=True)[1][0]

    # Split the input_ids at the index found, excluding -200
    input_ids_1 = input_ids[:, :split_index]
    input_ids_2 = input_ids[:, split_index + 1:]

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
    attention_mask = torch.ones(concatenated_embeddings.shape[:2], dtype=torch.long, device=device)
    return concatenated_embeddings, attention_mask


def initialize_models():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct", use_fast=True)
    model = LlamaForCausalLM.from_pretrained(
        "unsloth/llama-3-8b-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    for param in model.base_model.parameters():
        param.requires_grad = False

    model_name = "google/siglip-so400m-patch14-384"
    vision_model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
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
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.model(x)

def load_projection_module(mm_hidden_size=1152, hidden_size=4096, device='cuda'):
    projection_module = ProjectionModule(mm_hidden_size, hidden_size)
    checkpoint = torch.load("./checkpoints/llama-3/checkpoint-2400/mm_projector.bin")
    checkpoint = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}
    projection_module.load_state_dict(checkpoint)
    projection_module = projection_module.to(device).half()
    return projection_module


def answer_question(
        image_path, question, tokenizer, model, vision_model, processor, projection_module
):
    image = Image.open(image_path).convert('RGB')

    question = '<image>' + question

    wrap_sys = lambda msg: f"<|start_header_id|>system<|end_header_id|>\n\n{msg}<|eot_id|>" if len(msg) > 0 else msg
    wrap_user = lambda msg: f"<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>"
    wrap_assistant = lambda msg: f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg}<|eot_id|>"
    wrap_assistant_completion = lambda msg: f"<|start_header_id|>assistant<|end_header_id|>\n\n"

    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    print('QUESTION IS: ', prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).to(
        model.device)

    print('INPUT IDS SHAPE: ', input_ids.shape)
    print('INPUT IDS SHAPE: ', input_ids)

    with torch.inference_mode():
        image_inputs = processor(images=[image], return_tensors="pt", do_resize=True,
                                          size={"height": 384, "width": 384}).to("cuda")

        image_inputs = image_inputs['pixel_values'].squeeze(0)

        print('img inputs: ', image_inputs.shape)

        image_forward_outs = vision_model(image_inputs.to(device='cuda', dtype=torch.float16).unsqueeze(0),
                                               output_hidden_states=True)

        image_features = image_forward_outs.hidden_states[-2]

        print('IMG FORWARD OUT SHAPE: ', image_features)

        image_features2 = image_features[:, 1:]

        print('IMAGE FEATURES SHAPE BEFORE PROJ: ', image_features2.shape)

        projected_embeddings = projection_module(image_features2).to("cuda")

        print('IMAGE FEATURES SHAPE AFTER PROJ: ', projected_embeddings.shape)

        embedding_layer = model.get_input_embeddings()
        #text_embeddings = embedding_layer(input_ids)

        new_embeds, attn_mask = process_tensors(input_ids, projected_embeddings, embedding_layer)
        device = model.device
        attn_mask = attn_mask.to(device)
        new_embeds = new_embeds.to(device)
        print('MODEL DEVICE: ', device)

        print('NEW EMBEDS SHAPE: ', new_embeds.shape)
        print('ATTN MASK SHAPE: ', attn_mask.shape)
        print("ATTN MASK: ", attn_mask)

        model_kwargs = {
            'do_sample': True,
            'temperature': 0.2,
            'max_new_tokens': 500,
            'use_cache': True
        }

        generated_ids = model.generate(
            inputs_embeds=new_embeds,
            attention_mask=attn_mask,
            **model_kwargs

        )[0]
        response = tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_space=True
        )

        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer questions based on an image")
    parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    parser.add_argument("-q", "--question", required=True, help="Question to be asked")

    args = parser.parse_args()

    tokenizer, model, vision_model, processor = initialize_models()
    projection_module = load_projection_module()

    answer_question(
        args.image,
        args.question,
        tokenizer,
        model,
        vision_model,
        processor,
        projection_module,
    )
