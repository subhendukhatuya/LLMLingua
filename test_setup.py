from llmlingua import PromptCompressor

prompt_complex = open("./prompt_hardest.txt").read()
contxt = prompt_complex.split('\n\n')

llm_lingua = PromptCompressor()

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)

i = 0
dict_list = []
for prompt in contxt:
    
    i += 1
    compressed_prompt = llm_lingua.compress_prompt(prompt,)
    compressed_prompt2 = llm_lingua2.compress_prompt(prompt,)
    
    mydict = dict()
    mydict['Number'] = i
    mydict['LLMLingua Rate'] = compressed_prompt['rate']
    mydict['LLMLingua2 Rate'] = compressed_prompt2['rate']
    mydict['LLMLingua Compressed Prompt'] = compressed_prompt['compressed_prompt']
    mydict['LLMLingua2 Compressed Prompt'] = compressed_prompt2['compressed_prompt']

    dict_list.append(mydict)

import csv

with open('compare_llmlingua_1v2.csv', 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file, fieldnames=dict_list[0].keys(),)
    fc.writeheader()
    fc.writerows(dict_list)