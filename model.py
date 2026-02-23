import torch
import torch.nn as nn
import data_handling
import learning
pairs = [
    ("Dov'e la papera di gomma ?", "Where is the rubber duck ?"),
    ("Chi vi ha dato quella chitarra ?", "Who gave you that guitar ?"),
    ("Chi ti ha dato quel libro ?", "Who gave you that book ?"),
    ("Perche sei vegetariano ?", "Why are you a vegetarian ?"),
    ("Il suo discorso e stato splendido .", "Your speech was splendid ."),
    ("Tu riesci a catturare il pollo ?", "Can you catch the chicken ?"),
    ("Dove posso trovare un ristorante ?", "Where can I find a restaurant ?"),
    ("Mi puoi aiutare con i compiti ?", "Can you help me with my homework ?"),
    ("Quanto costa questo libro ?", "How much does this book cost ?"),
    ("Che ore sono adesso ?", "What time is it now ?"),
    ("Posso avere un bicchiere d'acqua ?", "Can I have a glass of water ?"),
    ("Hai visto il mio gatto ?", "Have you seen my cat ?"),
    ("Voglio imparare a suonare la chitarra .", "I want to learn to play the guitar ."),
    ("Dove si trova la stazione ferroviaria ?", "Where is the train station ?"),
    ("Come si chiama il tuo amico ?", "What is your friend's name ?")
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

training_model = learning.train(training_model,inputs, targets_one_hot, 10, 0.1, 100, 100, 100)

torch.save(training_model.state_dict(), "microtranslator.pth")
training_model.load_state_dict(torch.load("microtranslator.pth"))
outputs = training_model(inputs)

while True:
    sentence = input("Italian: ")

    if sentence.lower() == "quit":
        break

    print("Untrained:", predict(sentence, untrained_model))
    print("Trained:", predict(sentence, training_model))