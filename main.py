import os
import torch
from torch.utils.data import DataLoader
from dataset import PatientsDataset
from model import Classifier
from train import Trainer
from evaluate import Evaluator
import torch.optim as optim

def list_files(directory, extension=".tsv"):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    return files

def main():
    base_dir = os.getcwd()
    fmri_data_dir = os.path.join(base_dir, "fmri_data")
    metadata_dir = os.path.join(base_dir, "metadata")

    train_dir = os.path.join(fmri_data_dir, "train_tsv")
    test_dir = os.path.join(fmri_data_dir, "test_tsv")

    # tsv files
    train_files = list_files(train_dir)
    test_files = list_files(test_dir)

    # metadata files
    train_metadata_file = os.path.join(metadata_dir, "training_metadata.csv")
    test_metadata_file = os.path.join(metadata_dir, "test_metadata.csv")

    # create datasets
    train_dataset = PatientsDataset(train_files, train_metadata_file, is_train=True)
    test_dataset = PatientsDataset(test_files, test_metadata_file, is_train=False)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # initialize model, optimizer and loss function
    input_size = train_dataset[0][0].shape[0]
    num_classes = 1  # the age
    model = Classifier(input_size, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # decrease LR

    # Train the model
    trainer = Trainer(model, train_loader, optimizer, scheduler, num_epochs=10)
    trainer.train()

    # save the model after training
    torch.save(model.state_dict(), 'model.pth')

    # evaluate the model
    evaluator = Evaluator(model, test_loader, test_files)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
