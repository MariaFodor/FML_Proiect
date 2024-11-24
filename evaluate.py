import torch
import pandas as pd
import os

# class for evaluation
class Evaluator:
    def __init__(self, model, test_loader, test_files):
        self.model = model
        self.test_loader = test_loader
        self.test_files = test_files

    def evaluate(self):
        self.model.eval()
        predictions = []
        participant_ids = []

        with torch.no_grad():
            for i, inputs in enumerate(self.test_loader):
                outputs = self.model(inputs)
                predictions.extend(outputs.numpy().flatten())
                participant_ids.extend(self.test_files[i * 16:(i + 1) * 16])

        # write predictions in results.csv
        results = pd.DataFrame({
            "participant_id": [os.path.basename(f).split("_")[0] for f in participant_ids],
            "age": predictions
        })
        results.to_csv("results.csv", index=False)
        print("Predictions saved in results.csv.")
