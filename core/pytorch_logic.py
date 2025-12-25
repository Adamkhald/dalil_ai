import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class PyTorchPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = []
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history = {"loss": [], "acc": []}
        
    def load_data(self, data_dir, img_size=224, batch_size=32, val_split=0.2):
        """
        Loads data from ImageFolder structure.
        Expects: data_dir/class1, data_dir/class2...
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
            
        # Define Transforms
        data_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
        self.classes = full_dataset.classes
        
        # Split
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    def load_builtin_data(self, dataset_name, batch_size=32, val_split=0.2, img_size=224):
        """
        Loads built-in datasets (CIFAR10, MNIST, FashionMNIST).
        Downloads them to ./data directory if needed.
        """
        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Standard transforms
        # For MNIST (grayscale), we convert to 3 channels
        if dataset_name in ["MNIST", "FashionMNIST"]:
            t_list = [
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        else:
            # CIFAR10 etc are already 3 channels
            t_list = [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        data_transforms = transforms.Compose(t_list)
        
        try:
            if dataset_name == "CIFAR10":
                full_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_transforms)
                self.classes = full_dataset.classes
            elif dataset_name == "MNIST":
                full_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=data_transforms)
                self.classes = [str(i) for i in range(10)]
            elif dataset_name == "FashionMNIST":
                full_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=data_transforms)
                self.classes = full_dataset.classes
            else:
                return f"Dataset {dataset_name} not supported."
        except Exception as e:
            return f"Error downloading {dataset_name}: {e}"

        # Split
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return f"Loaded {dataset_name}. {len(full_dataset)} images. Classes: {self.classes}"

    def build_model(self, model_name="resnet18", num_classes=None, pretrained=True):
        if num_classes is None:
            num_classes = len(self.classes)
        
        if num_classes == 0:
            raise ValueError("No classes found. Load data first.")

        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "vgg16":
            self.model = models.vgg16(pretrained=pretrained)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.model = self.model.to(self.device)
        return f"Built {model_name} for {num_classes} classes on {self.device}."

    def setup_training(self, lr=0.001, optimizer_name="Adam"):
        if self.model is None:
            raise ValueError("Model not built.")
            
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            
    def train_one_epoch(self, epoch_idx, callback=None):
        if not self.train_loader:
            raise ValueError("Data not loaded.")
            
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Intra-epoch update every 10 batches or 10%
            if callback and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
                current_acc = correct / total
                current_loss = running_loss / total
                callback(batch_idx, total_batches, current_loss, current_acc)
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        
        self.history["loss"].append(epoch_loss)
        self.history["acc"].append(epoch_acc)
        
        return epoch_loss, epoch_acc
        
    def save_checkpoint(self, path="model_checkpoint.pth"):
        if self.model:
            torch.save(self.model.state_dict(), path)
            return f"Saved to {path}"
        return "No model to save."

    def run_validation(self, num_images=9):
        """
        Runs validation on a few images and returns them for visualization.
        Returns: List of dicts {'image': PIL_Image, 'true': str, 'pred': str}
        """
        if not self.val_loader: return []
        
        self.model.eval()
        results = []
        count = 0
        
        # Un-normalize for visualization
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for i in range(inputs.size(0)):
                    if count >= num_images: break
                    
                    # Process Image for display
                    img_tensor = inv_normalize(inputs[i]).cpu()
                    # Clip and Convert to PIL
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    from torchvision.transforms.functional import to_pil_image
                    pil_img = to_pil_image(img_tensor)
                    
                    true_label = self.classes[labels[i].item()]
                    pred_label = self.classes[preds[i].item()]
                    
                    results.append({"image": pil_img, "true": true_label, "pred": pred_label})
                    count += 1
                if count >= num_images: break
                
        return results

    def generate_inference_code(self, model_classname="resnet18", num_classes=10):
        """
        Generates a standalone Python script for inference.
        """
        code = f'''import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Config
MODEL_PATH = "model.pth"
NUM_CLASSES = {num_classes}
CLASSES = {self.classes}

# 2. Rebuild Model Architecture (Must match training)
def get_model():
    model = models.{model_classname}(pretrained=False)
    # Adjust final layer
    if "{model_classname}" == "resnet18":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif "{model_classname}" == "vgg16":
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    elif "{model_classname}" == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model

# 3. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 4. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)
        
    print(f"Prediction: {{CLASSES[preds[0].item()]}}")

if __name__ == "__main__":
    # Example usage
    # predict("test_image.jpg")
    print("Model loaded. Call predict(path) to use.")
'''
        return code
