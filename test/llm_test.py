import time
import sys
sys.path.append("..")

from controller.llm_wrapper import LLMWrapper

llm = LLMWrapper()

start = "Please generate the exact following content: "

a = "OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. You can use the tool below to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text. It's important to note that the exact tokenization process varies between models. Newer models like GPT-3.5 la and GPT-4 use a different tokenizer than our legacy GPT-3 and Codex models, and will produce different tokens for the same input text."

b = "[t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d, t, d]"
t1 = time.time()
response = llm.request(start + b)
t2 = time.time()
print(f"Time elapsed: {t2 - t1}, response: {response}")