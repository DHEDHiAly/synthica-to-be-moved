import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data import load_data, build_schema, build_sequences
from model import InvariantGRU


# ----------------------------
# batching utilities
# ----------------------------

def iterate_batches(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def batchify(batch):

    x = torch.nn.utils.rnn.pad_sequence(
        [b["x"] for b in batch],
        batch_first=True
    )

    y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
    h = torch.tensor([b["hospital"] for b in batch], dtype=torch.long)

    return x, y, h


# ----------------------------
# evaluation (task only)
# ----------------------------

def evaluate(model, data):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in iterate_batches(data, 32):

            x, y, _ = batchify(batch)

            y_pred, _ = model(x, alpha=0.0)

            preds.append(y_pred.cpu().numpy())
            labels.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return roc_auc_score(labels, preds)


# ----------------------------
# main training loop
# ----------------------------

def main():

    # ------------------------
    # load data
    # ------------------------
    df = load_data()
    schema = build_schema(df)
    seqs, _ = build_sequences(df, schema)

    np.random.shuffle(seqs)

    n = len(seqs)
    train = seqs[:int(0.7 * n)]
    val = seqs[int(0.7 * n):int(0.85 * n)]
    test = seqs[int(0.85 * n):]

    # ------------------------
    # model setup
    # ------------------------
    n_hospitals = max(b["hospital"] for b in seqs) + 1
    model = InvariantGRU(len(schema.features), n_hospitals=n_hospitals)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------
    # training
    # ------------------------
    for epoch in range(5):

        model.train()

        total_loss = 0.0

        for batch in iterate_batches(train, 32):

            x, y, h = batchify(batch)

            opt.zero_grad()

            y_pred, h_pred = model(x, alpha=1.0)

            # task loss (mortality prediction)
            task_loss = F.binary_cross_entropy_with_logits(
                y_pred,
                y
            )

            # adversarial loss (hospital prediction)
            hosp_loss = F.cross_entropy(h_pred, h)

            loss = task_loss + 0.3 * hosp_loss

            loss.backward()
            opt.step()

            total_loss += loss.item()

        val_auc = evaluate(model, val)

        print(
            f"Epoch {epoch} | "
            f"Loss: {total_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

    # ------------------------
    # final test
    # ------------------------
    test_auc = evaluate(model, test)

    print("\nFINAL TEST AUC:", test_auc)


if __name__ == "__main__":
    main()
