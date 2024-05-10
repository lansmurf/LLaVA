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


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def evaluate_model_and_save_csv(dataset, tokenizer, model, image_processor, output_file="evaluation_results_llama_3.csv"):
    results = []
    for example in tqdm(dataset):
        image = example['image']
        if isinstance(image, str):
            image = load_image(image)
        elif not isinstance(image, Image.Image):
            continue

        question = example['question']
        correct_answer = example['answer']

        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        with torch.inference_mode():
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_size = [image.size]

            model_kwargs = {
                'do_sample': False,
                'temperature': 0.2,
                'max_new_tokens': 2000,
                'use_cache': False,
                'images': image_tensor,
                'image_sizes': image_size,
            }

            output_ids = model.generate(input_ids, **model_kwargs)
            generated_text = tokenizer.decode(output_ids[0]).strip()
            print(generated_text)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load model and tokenizer using the first script's mechanism
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)


    # Load dataset
    dataset = load_dataset("xai-org/RealworldQA", split='test')

    # Run benchmarking
    evaluate_model_and_save_csv(dataset, tokenizer, model, image_processor)
