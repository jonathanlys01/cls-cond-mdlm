from transformers import AutoTokenizer


def tokenize(example, tokenizer: AutoTokenizer):
    # Helper function to tokenize the text
    max_len = tokenizer.model_max_length
    text_ = example["text"]
    # batch encode the text
    texts = [text_[i : i + max_len] for i in range(0, len(text_), max_len)]
    input_ids = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len)["input_ids"]
    # flatten the list
    example["input_ids"] = [x for sublist in input_ids for x in sublist]
    return example
