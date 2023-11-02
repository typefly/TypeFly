import time
import sys
sys.path.append("..")

from controller.llm_wrapper import LLMWrapper

llm = LLMWrapper()

start = "Please ignore the previous content and generate the exact following code: "

a = "OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. You can use the tool below to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text. It's important to note that the exact tokenization process varies between models. Newer models like GPT-3.5 la and GPT-4 use a different tokenizer than our legacy GPT-3 and Codex models, and will produce different tokens for the same input text. OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. You can use the tool below to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text. It's important to note that the exact tokenization process varies between models. Newer models like GPT-3.5 la and GPT-4 use a different tokenizer than our legacy GPT-3 and Codex models, and will produce different tokens for the same input text. OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. You can use the tool below to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text. It's important to note that the exact tokenization process varies between models. Newer models like GPT-3.5 la and GPT-4 use a different tokenizer than our legacy GPT-3 and Codex models, and will produce different tokens for the same input text. OpenAI's large language models (sometimes referred to as GPT's) process text using tokens, which are common sequences of characters found in a set of text. The models learn to understand the statistical relationships between these tokens, and excel at producing the next token in a sequence of tokens. You can use the tool below to understand how a piece of text might be tokenized by a language model, and the total count of tokens in that piece of text. It's important to note that the exact tokenization process varies between models. Newer models like GPT-3.5 la and GPT-4 use a different tokenizer than our legacy GPT-3 and Codex models, and will produce different tokens for the same input text."

b = "for i in range(4):\
    var_1 = obj_loc_y($1)\
    if var_1 > 0.6:\
        move_down(20)\
    if var_1 < 0.4:\
        move_up(20)\
    var_2 = obj_loc_y($1)\
    if var_2 < 0.6 and var_2 > 0.4:\
        return true\
return false"

c = "L 4\
  Var_1 obj_loc_y id=$1\
  C Var_1 > 0.6\
    S move_down value=20\
  C Var_1 < 0.4\
    S move_up value=20\
  Var_2 obj_loc_y id=$1\
  CR var_2 0.4 0.6 0.7 n\
    RT true\
RT false"

d = "Loop 4\n  Var_1 obj_loc_y id=$1\n  Cmp Var_1 > 0.6\n    S move_down value=20\n  Cmp Var_1 < 0.4\n    S move_up value=20\n  Var_2 obj_loc_y id=$1\n  CR var_2 0.4 0.6\n    RET true\nRET false"
t1 = time.time()
response = llm.request(start + d)
t2 = time.time()
print(f"Time elapsed: {t2 - t1}, response: {response}")
