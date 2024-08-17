from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def t5_small():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", model_max_length=128)
    transformer = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    return (tokenizer, transformer)

def t5_base():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", model_max_length=128)
    transformer = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    return (tokenizer, transformer)

def t5_large():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large", model_max_length=128)
    transformer = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large")
    return (tokenizer, transformer)