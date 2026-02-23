#!/usr/bin/env python3
"""
BlackRoad Labs — Minimal ML Training Pipeline
Demonstrates a reproducible training loop compatible with BlackRoad memory.
"""
import json
import time
import hashlib
import random
import math

class SimpleNN:
    """Two-layer MLP for demonstration."""
    def __init__(self, input_dim: int, hidden: int, output: int, lr: float = 0.01):
        self.lr = lr
        # He init
        self.W1 = [[random.gauss(0, math.sqrt(2/input_dim)) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[random.gauss(0, math.sqrt(2/hidden)) for _ in range(hidden)] for _ in range(output)]
        self.b2 = [0.0] * output

    def relu(self, x): return max(0.0, x)
    def drelu(self, x): return 1.0 if x > 0 else 0.0

    def forward(self, x):
        h = [self.relu(sum(self.W1[j][i]*x[i] for i in range(len(x))) + self.b1[j])
             for j in range(len(self.b1))]
        out = [sum(self.W2[k][j]*h[j] for j in range(len(h))) + self.b2[k]
               for k in range(len(self.b2))]
        return out, h

    def train_step(self, x, y_true):
        out, h = self.forward(x)
        loss = sum((out[k] - y_true[k])**2 for k in range(len(out))) / len(out)
        # Simple SGD backprop
        d_out = [(out[k] - y_true[k]) * 2 / len(out) for k in range(len(out))]
        for k in range(len(self.b2)):
            for j in range(len(h)):
                self.W2[k][j] -= self.lr * d_out[k] * h[j]
            self.b2[k] -= self.lr * d_out[k]
        return loss


class PSHashTracker:
    """PS-SHA∞ style training log — tamper-evident epoch history."""
    def __init__(self):
        self.chain: list[dict] = []
        self.prev_hash = "GENESIS"

    def log_epoch(self, epoch: int, loss: float, val_loss: float | None = None):
        content = json.dumps({"epoch": epoch, "loss": round(loss, 6), "val_loss": val_loss})
        ts = time.time_ns()
        h = hashlib.sha256(f"{self.prev_hash}:{content}:{ts}".encode()).hexdigest()
        entry = {"hash": h, "prev_hash": self.prev_hash, "epoch": epoch,
                 "loss": loss, "val_loss": val_loss, "timestamp_ns": ts}
        self.chain.append(entry)
        self.prev_hash = h
        return entry

    def verify(self) -> bool:
        ph = "GENESIS"
        for e in self.chain:
            content = json.dumps({"epoch": e["epoch"], "loss": round(e["loss"], 6), "val_loss": e["val_loss"]})
            expected = hashlib.sha256(f"{ph}:{content}:{e['timestamp_ns']}".encode()).hexdigest()
            if expected != e["hash"]:
                return False
            ph = e["hash"]
        return True


def generate_data(n=200):
    """XOR problem."""
    data = []
    for _ in range(n):
        x1, x2 = random.choice([0.0, 1.0]), random.choice([0.0, 1.0])
        y = float(int(x1) ^ int(x2))
        data.append(([x1, x2], [y]))
    return data


def train():
    print("BlackRoad Labs — ML Pipeline Demo")
    print("Problem: Learn XOR with a 2-layer MLP\n")

    data = generate_data(400)
    train_data, val_data = data[:300], data[300:]
    model = SimpleNN(input_dim=2, hidden=8, output=1, lr=0.05)
    tracker = PSHashTracker()

    for epoch in range(1, 51):
        random.shuffle(train_data)
        train_loss = sum(model.train_step(x, y) for x, y in train_data) / len(train_data)
        val_loss = sum(model.forward(x)[0][0] - y[0] for x, y in val_data) / len(val_data)
        val_loss = abs(val_loss)
        entry = tracker.log_epoch(epoch, train_loss, val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | hash={entry['hash'][:8]}")

    print(f"\nChain integrity: {'✓ VERIFIED' if tracker.verify() else '✗ BROKEN'}")
    print(f"Total epochs logged: {len(tracker.chain)}")

    # Test
    correct = sum(1 for x, y in val_data if round(model.forward(x)[0][0]) == round(y[0]))
    print(f"Accuracy on val: {correct}/{len(val_data)} = {correct/len(val_data)*100:.1f}%")


if __name__ == "__main__":
    train()
