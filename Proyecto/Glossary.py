from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"
# Continuation of the previous script
tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)
assert tokenized_sequence == ['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
# Continuation of the previous script
encoded_sequence = tokenizer.encode(sequence)
print(encoded_sequence)
assert encoded_sequence == [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
#   Attention Mask
