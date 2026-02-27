import torch
import torch.nn as nn
import data_handling
import learning
pairs = [
    ("Dov'e la papera ?", "Where is the duck ?"),
    ("Come si chiama il gatto ?", "What is the name of the cat ?"),
    ("Puoi aiutarmi catture il pollo ?", "Can you help me catch the chicken ?"),
    ("Perche sei vegetariano ?", "Why are you a vegetarian ?"),
    ("Il suo discorso e stato bello .", "Your speech was splendid ."),
    ("Hai visto il duomo","Have you seen the Duomo/Church"),
    ("Hai visto il gatto ?", "Have you seen cat ?"),
    ("Dov'e il lago ?", "Where is the lake ?")
]


vocab = data_handling.build_vocab(pairs)
vocab_size = len(vocab)
num_classes = len(pairs)
embed_dim = 16  # small is fine

encoded_inputs = [data_handling.encode_sentence(ita, vocab) for ita, _ in pairs]
max_len = max(len(seq) for seq in encoded_inputs)

padded_inputs = data_handling.pad_sequences(encoded_inputs, max_len)

inputs = torch.tensor(padded_inputs).long()
targets = torch.arange(num_classes).long()  # [0, 1, 2, 3, 4, 5]
targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()


class MicroTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        pooled_emb = emb.mean(dim=1)
        return self.fc(pooled_emb)


# Instantiate model
untrained_model = MicroTranslator()
training_model = MicroTranslator()



def predict(sentence,myModel):
    myModel.eval()

    tokens = data_handling.encode_sentence(sentence, vocab)

    # pad to max_len
    padded = tokens + [0] * (max_len - len(tokens))

    tensor_input = torch.tensor([padded]).long()

    with torch.no_grad():
        logits = myModel(tensor_input)
        predicted_class = torch.argmax(logits, dim=1).item()

    return pairs[predicted_class][1]

training_model = learning.train(training_model,inputs, targets_one_hot, 7, 0.9)

torch.save(training_model.state_dict(), "microtranslator.pth")
training_model.load_state_dict(torch.load("microtranslator.pth"))
outputs = training_model(inputs)

good_sentence = True

while True:
    sentence = input("Italian: ")

    if sentence.lower() == "quit":
        break
    else:
        for i in range(len(sentence.split())):
           if sentence.lower().split()[i] not in vocab:
               good_sentence = False
               print("Phrase not in vocab:", sentence.lower().split()[i])
               break
           else:
               good_sentence = True
        print(good_sentence)
        if good_sentence == True:
            print("Untrained:", predict(sentence, untrained_model))
            print("Trained:", predict(sentence, training_model))