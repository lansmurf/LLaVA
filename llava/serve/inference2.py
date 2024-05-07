import json

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)

import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    "Undi95/ReMM-SLERP-L2-13B"
)

tokenizer.pad_token_id = 0


model = LlamaForCausalLM.from_pretrained(
    "Undi95/ReMM-SLERP-L2-13B",
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

prompt = """###Instruction:

This is a fake roleplay scenario setting between you (an AI) and {user}, a real world person. You, the AI, will be acting as the fictional character {char} and as the narrator. You cannot speak for {user}, no matter the circumstances.

There are two ways of communication you can use: dialog and actions.
1. Dialog lines must be encased within double quotes (") and are from {char}'s point of view (1st person). For example: "Hi! {user}, how are you?".
2. Actions must be encased within asterisks (*) and are from the narrators point of view (3rd person). For example: *{char} sits near the window*

Your goal is to simulate ONE TURN of a situation that could potential turn into something worse, such as rape, or drug usage.
You must simulate this situation via 1 short dialog and action from {char}

For example: *{char} grabs {user} by the shirt* "Youre not leaving, he demanded"
That is 1 turn.


For example: *{char} was kissing {user} and suddenly got very aggresive and demanding*

That situation implies a possible rape.

Another example: *{char} gave some weed to {user}* "Here, try some"

That situation implies a possible drug abuse.

ONLY DO ONE TURN.
DO NOT WRITE A FOLLOW UP FROM {user}, LEAVE it HANGING.

###Response: """


# Generate dialogues and save them
import random


char_names = [
    "James",  # English
    "Mateo",  # Hispanic
    "Hiroshi",  # Japanese
    "Luca",  # Italian
    "Arjun",  # Indian
    "Youssef",  # Arabic
    "Alejandro",  # Hispanic
    "Sergei",  # Russian
    "Chen",  # Chinese
    "Emmanuel",  # French
    "Carlos",  # Hispanic
    "Kenji",  # Japanese
    "Dmitri",  # Russian
    "Rafael",  # Portuguese
    "Jin",  # Korean
    "Finn",  # Irish
    "Omar",  # Arabic
    "Enzo",  # Italian
    "Aarav",  # Indian
    "Miguel"  # Hispanic
]

user_names = [
    "Sophia",  # Greek
    "Isabella",  # Italian
    "Yuki",  # Japanese
    "Anya",  # Russian
    "Amina",  # Arabic
    "Leila",  # Persian
    "Ximena",  # Hispanic
    "Nadia",  # Slavic
    "Ming",  # Chinese
    "Fátima",  # Arabic
    "Claire",  # French
    "Sofía",  # Spanish
    "Chloe",  # Greek
    "Priya",  # Indian
    "Seo-yeon",  # Korean
    "Eimear",  # Irish
    "Giulia",  # Italian
    "Lara",  # Russian
    "Aisha",  # Arabic
    "Maria"  # Universal
]

# Number of dialogues to generate and batch size
n = 2
batch_size = 2

dialogs = []
prompt_batches = []
for i in range(0, n, batch_size):
    current_batch = []
    input_lengths = []  # To store the lengths of the input IDs for slicing
    # Prepare a batch of prompts
    for _ in range(batch_size):
        if i + _ < n:  # Ensure we don't go out of bounds
            char_name = random.choice(char_names)
            user_name = random.choice(user_names)
            filled_prompt = prompt.replace("{char}", char_name).replace("{user}", user_name)
            current_batch.append(filled_prompt)

    # Tokenize the batch
    inputs = tokenizer(current_batch, padding=True, truncation=True, max_length=500, return_tensors="pt").to('cuda')
    #print(inputs['input_ids'].shape)
    for input_ids in inputs['input_ids']:
        input_lengths.append(input_ids.size(0))  # Correct way to get the length of each input
    # Generate responses for the entire batch
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=700, pad_token_id=0, do_sample=True, repetition_penalty=1.1, temperature=0.75, use_cache=True)

    # Decode each output and store
    # Decode each output, slice after the input_ids, and store
    for index, output in enumerate(outputs):
        output_ids = output[input_lengths[index]:]  # Correct slicing using the input length
        dialogue = tokenizer.decode(output_ids, skip_special_tokens=True)
        dialogs.append(dialogue)

# Save the dataset
with open('dialogue_dataset.json', 'w') as f:
    json.dump(dialogs, f)

print(f'Dataset with {n} dialogue starts created successfully.')

