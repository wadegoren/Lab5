import os
import torch
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torch.nn as nn
import numpy as np
import time
from scipy.spatial.distance import cdist

class CustomTestDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.labels = []

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, 'test_noses.txt'), 'r') as file:
            for line in file:
                image_file, label, _ = line.strip().split(sep="\"")
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

def test(test_set, model, batch_size=1, pixel_range=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(args.w, map_location=device))
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    all_true_labels = []
    all_pred_labels = []

    image_test_time = []

    for data in test_loader:
        test_time_start = time.time()*1000
        inputs, labels, original_size = data

        resized_inputs = torch.nn.functional.interpolate(inputs, size=(original_size[1], original_size[0]),
                                                         mode='bilinear', align_corners=False)

        resized_inputs, labels = resized_inputs.to(device), labels.to(device)

        outputs = model(resized_inputs)
        all_true_labels.extend(labels.cpu().detach().numpy())
        all_pred_labels.extend(outputs.cpu().detach().numpy())
        test_time_end = time.time()*1000
        image_test_time.append(test_time_end - test_time_start)

    average_test_time = sum(image_test_time)/len(image_test_time)

    # Evaluate predictions based on pixel range
    correct_predictions = np.abs(np.array(all_true_labels) - np.array(all_pred_labels)) <= pixel_range
    accuracy = np.mean(correct_predictions)

    print(f"Accuracy within {pixel_range} pixels: {accuracy * 100:.2f}%")
    print(f"Average test time per image is: {average_test_time:.2f}")

    # Calculate Euclidean distances between true and predicted locations
    distances = cdist(np.array(all_true_labels), np.array(all_pred_labels))

    # Calculate localization accuracy statistics
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)

    print(f"Minimum distance: {min_distance:.2f} pixels")
    print(f"Mean distance: {mean_distance:.2f} pixels")
    print(f"Maximum distance: {max_distance:.2f} pixels")
    print(f"Standard deviation of distance: {std_distance:.2f} pixels")

    # Draw example pictures with predicted coordinates marked
    for i, (true_label, pred_label, is_correct) in enumerate(
            zip(all_true_labels, all_pred_labels, correct_predictions)):
        image, _, original_size = test_set[i]

        resized_image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)

        image_pil = transforms.ToPILImage()(resized_image.squeeze(0))
        draw = ImageDraw.Draw(image_pil)
        draw.ellipse([true_label[0] - pixel_range/10, true_label[1] - pixel_range/10,
                    true_label[0] + pixel_range/10, true_label[1] + pixel_range/10], outline="red")
        draw.ellipse([true_label[0] - pixel_range, true_label[1] - pixel_range,
                            true_label[0] + pixel_range, true_label[1] + pixel_range], outline="red")
        draw.ellipse([pred_label[0] - pixel_range, pred_label[1] - pixel_range,
                            pred_label[0] + pixel_range, pred_label[1] + pixel_range], outline="blue")
        plt.imshow(image_pil)
        plt.title(f"Example {i + 1}")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for test")
    parser.add_argument("-test_set", required=True, default="./data/images/",
                        help="Path to the content images directory")
    parser.add_argument("-b", type=int, default=32, help="Batch size for testing")
    parser.add_argument("-w", required=True, default="./model_weights.pth", help="Path to model weights")
    args = parser.parse_args()

    test_dataset_directory = args.test_set

    transform = transforms.Compose([
        transforms.Resize((400, 600)),
        transforms.ToTensor(),
    ])

    test_set = CustomTestDataset(directory=test_dataset_directory, transform=transform)

    resnet_model = resnet18(weights=None)
    resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)

    test(test_set, resnet_model, batch_size=args.b)
