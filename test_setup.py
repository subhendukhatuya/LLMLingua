from llmlingua import PromptCompressor

prompt_complex = open("./prompt_hardest.txt").read()
contxt = prompt_complex.split('\n\n')

llm_lingua = PromptCompressor()

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)

for prompt in contxt:
    compressed_prompt = llm_lingua.compress_prompt(
        prompt,
    )
    compressed_prompt2 = llm_lingua2.compress_prompt(
        prompt,
    )
    print(prompt)
    print()
    print("LLMLingua1")
    print(compressed_prompt)
    print()
    print("LLMLingua2")
    print(compressed_prompt2)



