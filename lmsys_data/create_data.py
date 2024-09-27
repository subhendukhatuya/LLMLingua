from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmlingua import PromptCompressor

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("lmsys/lmsys-chat-1m", cache_dir= '/NS/ssdecl/work/')

print('loaded')

# x =1/0
def create_message(row, content):
    """Create a message row based on the provided content."""
    st1 = time.time()
    compressed_prompt = llm_lingua2.compress_prompt(content,)
    compress_time = time.time()-st1
    return {
        'conversation_id': row['conversation_id'],
        'language': row['language'],
        'content': content,
        'compressed_prompt': compressed_prompt['compressed_prompt'],
        'time_taken': compress_time,
        'origin_tokens': compressed_prompt['origin_tokens'],
        'compressed_tokens': compressed_prompt['compressed_tokens'],
        'saving': compressed_prompt['saving']
    }

def process_conversation_data(initial_data):
    """Process the initial dataset to generate a new dataset."""
    processed_data = []

    for entry in tqdm(initial_data):
        user_messages = []
        assistant_messages = []
        turn_count = entry['turn']

        for msg in entry['conversation']:
            if msg['role'] == 'user':
                user_messages.append(msg['content'])
            elif msg['role'] == 'assistant':
                assistant_messages.append(msg['content'])

        for i in range(turn_count):
            combined_content = f"User: '{user_messages[i]}'"

            if i > 0:
                combined_content = []
                for j in range(i):
                    combined_content.append(f"User: '{user_messages[j]}'")
                    if j < len(assistant_messages):
                        combined_content.append(f"Assistant: '{assistant_messages[j]}'")

                combined_content = '\n'.join(combined_content) + f"\nUser: '{user_messages[i]}'"

            processed_data.append(create_message(entry, combined_content))

    return processed_data

def convert_to_dataset(processed_data):
    return Dataset.from_dict({key: [d[key] for d in processed_data] for key in processed_data[0]})

processed_splits = {}

for split in ds.keys():
    print(f"Processing {split} split")
    processed_data = process_conversation_data(ds[split].select(range(5)))
    processed_splits[split] = processed_data

new_dataset_dict = DatasetDict()

for split, data in processed_splits.items():
    new_dataset_dict[split] = Dataset.from_list(data)

# Save locally or push to Hugging Face Hub
# new_dataset_dict.save_to_disk("/content/processed_dataset_with_splits")
# OR
# new_dataset_dict.push_to_hub("mitramango/lmsys-processed-llmlingua2", private=True)
df = new_dataset_dict['train'].to_pandas()  
df.to_csv('sample_processed_lmsys_data.csv', index=False)
