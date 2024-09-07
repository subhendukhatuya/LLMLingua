from llmlingua import PromptCompressor

contxt = '''Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these
boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately
at the rate of three pens for $2. How much profit did he make in total, in dollars?
Let’s think step by step
Sam bought 12 boxes x $10 = $120 worth of highlighters.
He bought 12 * 30 = 360 highlighters in total.
Sam then took 5 boxes × 6 highlighters/box = 30 highlighters.
He sold these boxes for 5 * $3 = $15
After selling these 5 boxes there were 360 - 30 = 330 highlighters remaining.
These form 330 / 3 = 110 groups of three pens.
He sold each of these groups for $2 each, so made 110 * 2 = $220 from them.
In total, then, he earned $220 + $15 = $235.
Since his original cost was $120, he earned $235 - $120 = $115 in profit.
The answer is 115'''

llm_lingua = PromptCompressor()

compressed_prompt = llm_lingua.compress_prompt(
    contxt,
    target_token=60
)

llm_lingua2 = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)

compressed_prompt2 = llm_lingua2.compress_prompt(
    contxt,
    target_token=60
)

print(compressed_prompt)
print()
print(compressed_prompt2)