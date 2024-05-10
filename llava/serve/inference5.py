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
from transformers import default_data_collator


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def evaluate_model_and_save_csv(dataset, tokenizer, model, image_processor, batch_size=32,
                                output_file="evaluation_results_llama_3.csv"):
    results = []

    # Preparing the dataset in batches
    for batch_start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_start:batch_start + batch_size]
        questions = [example['question'] for example in batch]
        correct_answers = [example['answer'] for example in batch]
        images = [load_image(example['image']) if isinstance(example['image'], str) else example['image'] for example in
                  batch if isinstance(example['image'], (str, Image.Image))]

        # Skip batch if no valid images
        if not images:
            continue

        prompts = [f"<|start_header_id|>user<|end_header_id|>\n\n<image>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for question in questions]

        # Tokenize all prompts at once
        input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=tokenizer.pad_token_id).to(model.device)

        # Process all images at once
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = image_tensors.to(model.device, dtype=torch.float16)
        image_sizes = [image.size for image in images]

        with torch.inference_mode():
            model_kwargs = {
                'do_sample': False,
                'temperature': 0.2,
                'max_new_tokens': 100,
                'use_cache': True,
                'images': image_tensors,
                'image_sizes': image_sizes,
            }

            output_ids = model.generate(input_ids, **model_kwargs)

            # Decode each item in the batch
            for idx, output_id in enumerate(output_ids):
                generated_text = tokenizer.decode(output_id, skip_special_tokens=True).strip()
                results.append({
                    'Image Path': batch[idx]['image'] if isinstance(batch[idx]['image'],
                                                                    str) else 'Image loaded directly',
                    'Question': questions[idx],
                    'Correct Answer': correct_answers[idx],
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
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit,
                                                                           device=args.device)

    # Load dataset
    dataset = load_dataset("xai-org/RealworldQA", split='test')

    # Run benchmarking
    evaluate_model_and_save_csv(dataset, tokenizer, model, image_processor, batch_size=args.batch_size)
