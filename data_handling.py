from collections import Counter

#  Create our vocab, taking training pairs as input
def build_vocab(pairs):
    counter = Counter()

    for ita, _ in pairs:
        tokens = tokenize(ita)
        counter.update(tokens)

    vocab = {word: idx+1 for idx, word in enumerate(counter.keys())}
    vocab["<PAD>"] = 0
    print(vocab)

    return vocab
def tokenize(sentence):
    return sentence.lower().split()

def encode_sentence(sentence, vocab):
    tokens = tokenize(sentence)
    return [vocab[token] for token in tokens]

def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        seq = seq + [0] * (max_len - len(seq))
        padded.append(seq)
    return padded