import torch.nn as nn
import torch


class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance and focus on hard examples."""
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class ProteinStructurePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProteinStructurePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.layer4(x)
        return x


def train_model(model, train_data, train_labels, epochs=100):
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    losses = []

    for epoch in range(epochs):
        model.train()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Adjust learning rate
        losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return losses


def predict_structure(model, input_data):
    model.eval()
    with torch.no_grad():
        return model(input_data)
