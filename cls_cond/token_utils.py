import time

from transformers import AutoTokenizer


def chunked_tokenize(example, tokenizer: AutoTokenizer, max_len = None):
    # Faster (chunked) tokenization
    if max_len is None:
        max_len = tokenizer.model_max_length
    text_ = example["text"]
    assert isinstance(text_, str), "Input must be a string"
    # batch encode the text
    texts = [text_[i : i + max_len] for i in range(0, len(text_), max_len)] # size in chars < size in tokens
    input_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    # flatten the list
    example["input_ids"] = [x for sublist in input_ids for x in sublist]
    return example

def test_compare():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    example = {"text": "This is a long text. " * 1_000_000}

    # Tokenize the whole text at once
    start = time.time()
    ids = tokenizer(example["text"])["input_ids"]
    print("Time for whole text:", time.time() - start)

    # Tokenize the text in chunks
    start = time.time()
    example = chunked_tokenize(example, tokenizer)
    print("Time for chunks:", time.time() - start)

    print("Equal ids?", ids == example["input_ids"])
    # this might be False, due to BPE tokenization

    text = tokenizer.decode(ids)
    text_chunked = tokenizer.decode(example["input_ids"])

    print("Equal text?", text == text_chunked)
    # this should be True



if __name__ == "__main__":
    test_compare()



