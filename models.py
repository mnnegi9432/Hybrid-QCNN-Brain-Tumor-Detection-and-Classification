import torch
import torch.nn as nn


class HybridQCNN(nn.Module):
    def __init__(self, quantum_layer):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.quantum = quantum_layer

        self.classifier = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = self.embedding(x)

        outputs = []
        for sample in x:
            outputs.append(self.quantum(sample))

        x = torch.stack(outputs)

        return self.classifier(x)
