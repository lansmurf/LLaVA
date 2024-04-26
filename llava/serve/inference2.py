import argparse
import sys
import os.path as path
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from transformers import TextStreamer


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
    checkpoint = torch.load("./mm_projector.bin")
    checkpoint = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}
    projection_module.load_state_dict(checkpoint)
    projection_module = projection_module.to(device).half()
    return projection_module


def get_image_inputs(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(
        images=[image],
        return_tensors="pt",
        do_resize=True,
        size={"height": 384, "width": 384},
    ).to("cuda")
    return image_inputs["pixel_values"].squeeze(0)


@torch.inference_mode()
def get_embeds(prompt, vision_model, processor, projection_module, tokenizer, image_path, first=True):
    tokenizer.eos_token = "<|eot_id|>"

    device = model.device
    input_ids = (
        tokenizer_image_token(prompt, tokenizer)
        .unsqueeze(0)
        .to(model.device)
    )

    embedding_layer = model.get_input_embeddings()

    # if this isn't the first message
    if not first:
        new_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            device
        )
        new_embeddings = embedding_layer(new_input_ids)

        new_embeds = new_embeddings.to(device)
        attn_mask = torch.ones(new_embeds.shape[:2], device=device)

        return new_embeds, attn_mask

    # assume this is first msg so process img.
    image_inputs = get_image_inputs(image_path, processor)

    image_forward_outs = vision_model(
        image_inputs.to(device="cuda", dtype=torch.float16).unsqueeze(0),
        output_hidden_states=True,
    )

    image_features = image_forward_outs.hidden_states[-2]

    projected_embeddings = projection_module(image_features).to("cuda")

    # text_embeddings = embedding_layer(input_ids)

    new_embeds, attn_mask = process_tensors(
        input_ids, projected_embeddings, embedding_layer
    )
    attn_mask = attn_mask.to(device)
    new_embeds = new_embeds.to(device)

    return new_embeds, attn_mask


@torch.inference_mode()
def answer_question(
    tokenizer, model, new_embeds, attn_mask
):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model_kwargs = {
        "do_sample": True,
        "temperature": 0.2,
        "max_new_tokens": 2000,
        "use_cache": True,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id
    }

    generated_ids = model.generate(
        inputs_embeds=new_embeds, attention_mask=attn_mask, **model_kwargs
    )[0]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    return generated_text


@torch.inference_mode()
def chat_inference(
    image_path, tokenizer, model, vision_model, processor, projection_module
):
    image_path = image_path
    generated_text = ""
    while True:
        try:
            q = input("\nuser: ")
        except EOFError:
            q = ""
        if not q:
            print("no input detected. exiting.")
        elif q == "/help":
            print(
                "Type your message to the assistant.\n\
        Type '/exit' to exit the chat.\n\
        Type '/clear' to clear the chat history.\n\
        Type '/image <path>' to change the image.\n\
        Type '/help' to display this message."
            )
            continue
        elif q == "/exit":
            print("exiting.")
            sys.exit()
        elif q == "/clear":
            print("clearing chat history.")
            generated_text = ""
            q = ""
            continue
        elif q.startswith("/image"):
            # clear context so that image is first
            generated_text = ""
            # path is everything after "/image "
            image_path = q[len("/image") + 1:]
            if not path.exists(image_path):
                print("Image path does not exist.\n")
                continue

            continue

        if generated_text:
            prompt = (
                generated_text
                + "<|start_header_id|>user<|end_header_id|>\n\n"
                + q
                + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            new_embeds, attn_mask = get_embeds(prompt, vision_model, processor, projection_module, tokenizer,
                                               image_path, first=False)
            text = answer_question(tokenizer, model, new_embeds, attn_mask)

        else:
            question = "<image>" + q
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            new_embeds, attn_mask = get_embeds(prompt, vision_model, processor, projection_module, tokenizer,
                                               image_path)
            text = answer_question(tokenizer, model, new_embeds, attn_mask)

        generated_text += text
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer questions based on an image")
    parser.add_argument("-i", "--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    tokenizer, model, vision_model, processor = initialize_models()
    projection_module = load_projection_module()

    chat_inference(
        args.image,
        tokenizer,
        model,
        vision_model,
        processor,
        projection_module,
    )
