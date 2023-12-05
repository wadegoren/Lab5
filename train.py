import os
import torch
import argparse
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.labels = []

        # Read the labels.txt file
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, 'train_noses.2.txt'), 'r') as file:
            for line in file:
                image_file, label = line.strip().split(sep="\"")[:2]
                image_file = image_file[:-1]
                coordinates = label.strip('()').split(', ')
                x, y = map(int, coordinates)
                self.labels.append((image_file, int(x), int(y)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_file, x, y = self.labels[idx]
        image_path = os.path.join(self.directory, image_file)
        image = Image.open(image_path).convert('RGB')

        original_size = image.size
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([x, y], dtype=torch.float32), original_size

def train(train_set, batch_size, num_epochs, pth, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}...')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust step_size and gamma as needed

    training_loss = []

    for epoch in range(num_epochs):
        model.train()
        batch_loss = 0
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f'Current Time: {current_time}, Starting Epoch {epoch + 1} out of {num_epochs}')

        for data in train_loader:
            inputs, labels, original_size = data

            resized_inputs = torch.nn.functional.interpolate(inputs, size=(original_size[1], original_size[0]),
                                                mode='bilinear', align_corners=False)

            resized_inputs, labels = resized_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(resized_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            batch_loss += loss.item()

        scheduler.step()  # Step the learning rate scheduler

        epoch_loss = batch_loss / len(train_loader)
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f'Current Time: {current_time}, Epoch {epoch + 1} out of {num_epochs}, Loss: {epoch_loss}')
        training_loss.append(epoch_loss)
        torch.save(model.state_dict(), pth)

    torch.save(model.state_dict(), pth)

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend()
    plt.savefig("LossCurve")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for train")
    parser.add_argument("-train_set", required=True, default="/data/images/", help="Path to the content images directory")
    parser.add_argument("-e", type=int, default=20, help="Number of training epochs")
    parser.add_argument("-b", type=int, default=32, help="Batch size for training")
    parser.add_argument("-s", required=True, help="Path to save the decoder model")
    parser.add_argument("-cuda", choices=("Y", "N"), default="Y", help="Use CUDA for training (Y/N)")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((400, 600)),  # Adjust size as needed
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    train_set = CustomDataset(directory=args.train_set, transform=transform)
    resnet_model = resnet18(weights=None)
    # train_set, batch_size, num_epochs, cuda
    train(train_set, args.b, args.e, args.s, resnet_model)
