## Setup
```
pip install -r requirements.txt
```

## To compress lmsys-data and push to hf-hub (already pushed)

```
cd lmsys_data
python create_data.py
```

## To compress arxiv-data and push to hf-hub (not pushed, also please change HF profile to the organization instead of mitramango)

```
cd arxiv_data
python create_arxiv_data.py --action push_hf
```

## To test compression results on arxiv-data (5 examples) and save csv

```
cd arxiv_data
python create_arxiv_data.py --action save_csv
```

## To use LLMLingua2 for compression (Python Code)
```
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmlingua import PromptCompressor

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
) # for LLMLingua2-large

llm_lingua2_small = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,  # Whether to use llmlingua-2
) # for LLMLingua2-small

prompt = "Input your prompt here"

compress_results = llm_lingua2.compress_prompt(prompt)
```