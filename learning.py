import torch


def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()

def train(model, inputs, targets, epochs, rate_learning, embedding, weights, biases):
    for i in range(epochs):
        # Print and update
        outputs = model(inputs)
        if i % 10 == 0:
            print(f"Epoch {epochs}")
        loss = []
        probs = []
        dl_dz = []
        h = []
        for v in range(outputs.size(0)):  # loop over examples
            probs_v = softmax(outputs[v])  # shape: (num_classes,)
            loss_v = -torch.sum(targets[v] * torch.log(probs_v))  # elementwise multiply + sum

            dl_dz_v = probs_v - targets[v]

            emb_v = model.embedding(inputs[v])
            h_v = emb_v.mean(0)

            dl_dh_v = model.fc.weight.T @ dl_dz_v
            seq_len = inputs[v].size(0)
            dl_demb_v = dl_dh_v / seq_len
            for t in range(seq_len):
                model.embedding.weight.data[inputs[v, t]] -= rate_learning * dl_demb_v

            h.append(h_v)
            dl_dz.append(dl_dz_v)
            loss.append(loss_v)
            probs.append(probs_v)

        dl_dz = torch.stack(dl_dz)
        h = torch.stack(h)

        dl_dw = dl_dz.T @ h
        dl_db = dl_dz.sum(0)



        with torch.no_grad():
            model.fc.weight -= rate_learning * dl_dw
            model.fc.bias -= rate_learning * dl_db
        total_loss = torch.mean(torch.stack(loss))
        print("Epoch", i, "loss:", total_loss.item())



    return model
