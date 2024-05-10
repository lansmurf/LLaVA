import argparse
import torch
from PIL import Image
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import pandas as pd
from datasets import load_dataset
from io import BytesIO
import requests

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def evaluate_model_and_save_csv(dataset, tokenizer, model, image_processor, batch_size=32, output_file="evaluation_results_llama_3.csv"):
    results = []

    # Initialize batch accumulators
    batch_images = []
    batch_questions = []
    batch_correct_answers = []

    # Iterate through each example in the dataset
    for example in tqdm(dataset):
        image = example['image']
        if isinstance(image, str):
            image = load_image(image)
        elif not isinstance(image, Image.Image):
            continue

        batch_images.append(image)
        batch_questions.append(example['question'])
        batch_correct_answers.append(example['answer'])

        # Check if the batch is full
        if len(batch_images) == batch_size:
            # Process the full batch
            prompts = [f"<|start_header_id|>user<|end_header_id|>\n\n<image>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for question in batch_questions]
            input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)

            print(input_ids.shape)
            # Process all images at once
            image_tensors = process_images(batch_images, image_processor, model.config)
            image_tensors = image_tensors.to(model.device, dtype=torch.float16)
            image_sizes = [image.size for image in batch_images]

            print(image_tensors.shape)
            print(image_sizes)

            with torch.inference_mode():
                model_kwargs = {
                    'do_sample': False,
                    'temperature': 0.2,
                    'max_new_tokens': 2000,
                    'use_cache': True,
                    'images': image_tensors,
                    'image_sizes': image_sizes,
                }

                output_ids = model.generate(input_ids, **model_kwargs)

                # Decode each item in the batch
                for idx, output_id in enumerate(output_ids):
                    generated_text = tokenizer.decode(output_id, skip_special_tokens=True).strip()
                    results.append({
                        'Image Path': batch_images[idx] if isinstance(batch_images[idx], str) else 'Image loaded directly',
                        'Question': batch_questions[idx],
                        'Correct Answer': batch_correct_answers[idx],
                        'Generated Answer': generated_text
                    })

            # Clear the batch accumulators
            batch_images = []
            batch_questions = []
            batch_correct_answers = []

    # Process the last batch if it's not empty and less than batch_size
    if batch_images:
        prompts = [f"user\n\n<image>{question}assistant\n\n" for question in batch_questions]
        input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(model.device)

        # Process all images at once
        image_tensors = process_images(batch_images, image_processor, model.config)
        image_tensors = image_tensors.to(model.device, dtype=torch.float16)
        image_sizes = [image.size for image in batch_images]

        with torch.inference_mode():
            model_kwargs = {
                'do_sample': False,
                'temperature': 0.2,
                'max_new_tokens': 2000,
                'use_cache': True,
                'images': image_tensors,
                'image_sizes': image_sizes,
            }

            output_ids = model.generate(input_ids, **model_kwargs)

            # Decode each item in the batch
            for idx, output_id in enumerate(output_ids):
                generated_text = tokenizer.decode(output_id, skip_special_tokens=True).strip()
                results.append({
                    'Image Path': batch_images[idx] if isinstance(batch_images[idx], str) else 'Image loaded directly',
                    'Question': batch_questions[idx],
                    'Correct Answer': batch_correct_answers[idx],
                    'Generated Answer': generated_text
                })

    # Create a DataFrame and write to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Load model and tokenizer using the first script's mechanism
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # Load dataset
    dataset = load_dataset("xai-org/RealworldQA", split='test')

    # Run benchmarking
    evaluate_model_and_save_csv(dataset, tokenizer, model, image_processor, batch_size=args.batch_size)
