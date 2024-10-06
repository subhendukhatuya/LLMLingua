import argparse
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmlingua import PromptCompressor

# Initialize LLMLingua2 Prompt Compressor
llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)

# Argument parser to choose between processing first 5 samples or pushing full dataset to Hugging Face
parser = argparse.ArgumentParser(description="Process and compress arXiv dataset")
parser.add_argument('--action', type=str, choices=['save_csv', 'push_hf'], required=True,
                    help="Choose 'save_csv' to process and save first 5 samples to CSV, or 'push_hf' to process the full dataset and push to Hugging Face.")
args = parser.parse_args()

# Load the CSV file with sample numbers (0-indexed)
csv_file = 'arxiv_4k.csv'
df = pd.read_csv(csv_file)

# Extract the sample numbers (for first 5 if save_csv, otherwise all for push_hf)
sample_indices = df['sample_numbers'].tolist()

# Load the 'ccdv/arxiv-summarization' dataset
ds = load_dataset('ccdv/arxiv-summarization')

# Function to compress an article
def compress_article(article):
    st1 = time.time()
    compressed_result = llm_lingua2.compress_prompt(article)
    compress_time = time.time() - st1
    return {
        'compressed_article': compressed_result['compressed_prompt'],
        'compression_time': compress_time,
        'original_tokens': compressed_result['origin_tokens'],
        'compressed_tokens': compressed_result['compressed_tokens'],
        'token_saving': compressed_result['saving']
    }

# Process and compress the selected samples
processed_data = []
def process_samples(indices):
    for idx in tqdm(indices):
        row = ds['train'][idx]
        compressed_info = compress_article(row['article'])
        processed_data.append({
            'sample_number': idx,
            'abstract': row['abstract'],
            'article': row['article'],
            **compressed_info
        })

# Action: Save the first 5 processed examples to CSV
if args.action == 'save_csv':
    process_samples(sample_indices[:5])  # Process only the first 5 samples
    first_5_examples = pd.DataFrame(processed_data)
    first_5_examples.to_csv('first_5_compressed_examples.csv', index=False)
    print("First 5 processed examples saved to 'first_5_compressed_examples.csv'!")

# Action: Process and push the entire dataset to Hugging Face
elif args.action == 'push_hf':
    process_samples(sample_indices)  # Process all samples
    new_dataset = Dataset.from_dict({key: [d[key] for d in processed_data] for key in processed_data[0]})
    new_dataset_dict = DatasetDict({'train': new_dataset})

    # Push the dataset to Hugging Face Hub (make sure to login using `huggingface-cli login`)
    new_dataset_dict.push_to_hub("mitramango/arxiv-compressed", private=True)
    print("Full dataset processed and uploaded to Hugging Face!")
