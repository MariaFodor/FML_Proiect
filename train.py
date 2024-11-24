import torch

# class for training
class Trainer:
    def __init__(self, model, train_loader, optimizer, scheduler, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

    def compute_rmse(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_rmse = 0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.compute_rmse(outputs, labels.view(-1, 1))  # RMSE Loss Calculation
                loss.backward()
                self.optimizer.step()

                # get RMSE
                rmse = loss.item()
                total_rmse += rmse
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, RMSE: {total_rmse / len(self.train_loader):.4f}")
