from llmlingua import PromptCompressor

contxt = '''Q: I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many
fruits do I have?
A: Let’s think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- blackberry (1) - nectarine (1) - plum (1) - strawberry (1) - banana (1) - orange (1)
Now, let’s add the numbers in parentheses: 1 + 1 + 1 + 1 + 1 + 1 = 6. So the answer is 6.'''

llm_lingua = PromptCompressor()

compressed_prompt = llm_lingua.compress_prompt(
    contxt
)

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)

compressed_prompt2 = llm_lingua2.compress_prompt(
    contxt    
)

print(compressed_prompt)
print()
print(compressed_prompt2)