from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmlingua import PromptCompressor

llm_lingua2_small = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,  # Whether to use llmlingua-2
)

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,  # Whether to use llmlingua-2
)

import pandas as pd

# Initial compression of the prompt 'my name is mm' using both models
initial_prompt = 'my name is mm'

# Compress using llm_lingua2_small
start_time_small_init = time.time()
compressed_prompt_small_init = llm_lingua2_small.compress_prompt(initial_prompt)
end_time_small_init = time.time()

# Log time taken for llm_lingua2_small
time_taken_small_init = end_time_small_init - start_time_small_init
print(f"Initial prompt 'my name is mm': Time taken by llm_lingua2_small: {time_taken_small_init:.4f} seconds")
print(f"Compressed by llm_lingua2_small: {compressed_prompt_small_init}")

# Compress using llm_lingua2
start_time_large_init = time.time()
compressed_prompt_large_init = llm_lingua2.compress_prompt(initial_prompt)
end_time_large_init = time.time()

# Log time taken for llm_lingua2
time_taken_large_init = end_time_large_init - start_time_large_init
print(f"Initial prompt 'my name is mm': Time taken by llm_lingua2: {time_taken_large_init:.4f} seconds")
print(f"Compressed by llm_lingua2: {compressed_prompt_large_init}")

# Now, load the dataset and process the 7th row
df = pd.read_csv('sample_processed_lmsys_data.csv')

# Ensure the DataFrame has a 'content' column and at least 7 rows
if 'content' in df.columns and len(df) >= 7:
    # Select the 'content' from the 7th row (index 6)
    prompt = df.loc[6, 'content']

    # Time and compress using llm_lingua2_small
    start_time_small = time.time()
    compressed_prompt_small = llm_lingua2_small.compress_prompt(prompt)
    end_time_small = time.time()

    # Log time taken for llm_lingua2_small
    time_taken_small = end_time_small - start_time_small
    print(f"7th Row (Content): Time taken by llm_lingua2_small: {time_taken_small:.4f} seconds")
    
    # Time and compress using llm_lingua2
    start_time_large = time.time()
    compressed_prompt_large = llm_lingua2.compress_prompt(prompt)
    end_time_large = time.time()

    # Log time taken for llm_lingua2
    time_taken_large = end_time_large - start_time_large
    print(f"7th Row (Content): Time taken by llm_lingua2: {time_taken_large:.4f} seconds")

    # Output compressed results
    print(f"Compressed by llm_lingua2_small: {compressed_prompt_small}")
    print(f"Compressed by llm_lingua2: {compressed_prompt_large}")
else:
    print("The dataset must have a 'content' column and at least 7 rows.")
