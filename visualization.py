import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd

# Define the path for saving the model
MODEL_PATH = 'eurosat_autoencoder.pth'

# --- 1. Configuration and Data Loading Setup ---

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # Normalize with EuroSAT standard values (Crucial for Autoencoder performance)
    transforms.Normalize(mean=[0.3444, 0.3804, 0.4287], std=[0.2312, 0.1873, 0.1610])
])

# Specify the root directory for the dataset
# NOTE: Update this path to your actual EuroSAT data location
DATA_ROOT = '/Users/vejandlaanji/Desktop/EuroSat_data/eurosat_data/EuroSAT' 

# Load EuroSAT
print(f"Loading/downloading EuroSAT data from: {DATA_ROOT}")
try:
    # Set download=False if data is guaranteed to be there
    full_dataset = datasets.EuroSAT(root=DATA_ROOT, transform=transform, download=False)
except Exception as e:
    print(f"Error loading EuroSAT dataset (Is DATA_ROOT correct?): {e}")
    sys.exit(1)

# Select a "Normal" class (e.g., 'PermanentCrop' - class 4)
NORMAL_CLASS_LABEL = 4
NORMAL_CLASS_NAME = full_dataset.classes[NORMAL_CLASS_LABEL]

# Filter data to create a 'Normal' training set and a mixed 'Test' set
normal_indices = [i for i, (_, label) in enumerate(full_dataset) if label == NORMAL_CLASS_LABEL]
anomaly_indices = [i for i, (_, label) in enumerate(full_dataset) if label != NORMAL_CLASS_LABEL]

# Split 'Normal' data into training (80%) and test (20%)
# The test_normal_indices will be mixed with anomaly_indices later
train_normal_indices, test_normal_indices = train_test_split(
    normal_indices, test_size=0.2, random_state=42
)

# Create final Test set: a mix of normal and anomalous samples
test_indices = test_normal_indices + anomaly_indices 

# Create custom datasets
train_dataset = Subset(full_dataset, train_normal_indices)
test_dataset = Subset(full_dataset, test_indices)

# Define Data Loaders
BATCH_SIZE = 128 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("--- Data Setup Complete ---")
print(f"Normal Class: {NORMAL_CLASS_NAME}")
print(f"Training Samples (Normal Only): {len(train_dataset)}")
print(f"Test Samples (Normal + Anomaly): {len(test_dataset)}")


# --- 2. Define the Anomaly Detection Model (Simple Autoencoder) ---

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (3x64x64 input -> latent space)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), # 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 8x8
            nn.ReLU()
        )
        
        # Decoder (latent space -> 3x64x64 output)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), # 64x64
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 3. Training the Autoencoder on ONLY Normal Data ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleAutoencoder().to(DEVICE)
criterion = nn.MSELoss() # Reconstruction Loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
NUM_EPOCHS = 20
loss_history = [] # LIST TO STORE EPOCH LOSSES FOR PLOTTING

print(f"\n--- Starting Training on {DEVICE} for {NUM_EPOCHS} epochs---")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for data, _ in train_loader:
        # Input (data) is its own target for reconstruction
        data = data.to(DEVICE) 
        
        optimizer.zero_grad()
        reconstructions = model(data)
        loss = criterion(reconstructions, data)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    loss_history.append(epoch_loss) # STORE THE LOSS
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.6f}")

print("--- Training Complete ---")

# --- 4. Export the Trained Model ---
try:
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n*** Anomaly Detection Model successfully saved to: {MODEL_PATH} ***")
except Exception as e:
    print(f"Error saving model: {e}")
    
# --- 5. Anomaly Detection Evaluation (Calculate Scores) ---

def evaluate_anomaly_scores(loader, model, device):
    """Calculates the reconstruction error (anomaly score) for each image."""
    model.eval()
    scores = []
    labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            reconstructions = model(data)
            
            # Calculate the MSE per image (Anomaly Score) across all pixels and channels
            batch_scores = torch.mean((reconstructions - data)**2, dim=[1, 2, 3])
            
            scores.extend(batch_scores.cpu().numpy().tolist())
            labels.extend(target.cpu().numpy().tolist())
            
    return np.array(scores), np.array(labels)

anomaly_scores, test_labels = evaluate_anomaly_scores(test_loader, model, DEVICE)

# Convert actual labels to binary: 0 = Normal, 1 = Anomaly
binary_labels = np.array([0 if label == NORMAL_CLASS_LABEL else 1 for label in test_labels])

# Calculate AUC-ROC (standard metric for anomaly detection)
roc_auc = roc_auc_score(binary_labels, anomaly_scores)

print("\n--- Anomaly Detection Evaluation ---")
print(f"Normal Class: {NORMAL_CLASS_NAME}")
print(f"Anomaly Detection AUC-ROC: {roc_auc:.4f}")

# --- 6. Visualization and Diagrammatic Results ---
print("\n--- Generating Visualizations (Loss, Heatmap, Confusion Matrix) ---")

# --- A. Loss Curve ---
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss', color='darkblue')
plt.title('Autoencoder Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Mean Squared Error (Reconstruction Loss)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png')
print("Loss curve saved as 'loss_curve.png'")
plt.close()

# --- B. Anomaly Heatmap (Reconstruction Error Visualization) ---
def visualize_reconstruction_error(dataset: Subset, model: nn.Module, device: torch.device) -> None:
    """Plots original image, reconstruction, and the squared error heatmap."""
    model.eval()
    
    # Select a random anomalous image from the test set for a good example
    anomaly_indices_in_test = [i for i, label in enumerate(test_labels) if label != NORMAL_CLASS_LABEL]
    if not anomaly_indices_in_test:
        print("No anomalous samples found in the test set to visualize.")
        return

    # Use a random anomaly for visualization
    sample_idx = random.choice(anomaly_indices_in_test)
    
    # Get the data from the Subset (returns tuple of data and label index)
    original_data, original_label = dataset[sample_idx]
    
    # Denormalization parameters (reverse the transformation)
    mean = torch.tensor([0.3444, 0.3804, 0.4287]).view(3, 1, 1)
    std = torch.tensor([0.2312, 0.1873, 0.1610]).view(3, 1, 1)
    
    with torch.no_grad():
        input_data = original_data.unsqueeze(0).to(device)
        reconstruction = model(input_data).squeeze(0).cpu()
        
    # Denormalize the tensors for visualization (clamp to [0, 1] for display)
    original_denorm = (original_data * std + mean).clamp(0, 1)
    reconstruction_denorm = (reconstruction * std + mean).clamp(0, 1)
    
    # Calculate squared error (per pixel, averaged over channels)
    error_map = torch.mean((reconstruction - original_data.cpu())**2, dim=0) # Error map is 64x64
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original Image
    axes[0].imshow(original_denorm.permute(1, 2, 0))
    axes[0].set_title(f'Original Image\n(True Label: {full_dataset.classes[original_label]})')
    axes[0].axis('off')

    # 2. Reconstruction
    axes[1].imshow(reconstruction_denorm.permute(1, 2, 0))
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')

    # 3. Heatmap of Reconstruction Error
    im = axes[2].imshow(error_map.numpy(), cmap='inferno')
    axes[2].set_title('Reconstruction Error Heatmap\n(Anomaly Score)')
    axes[2].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Anomaly Detection Visualization for a Sample (Score: {anomaly_scores[sample_idx]:.6f})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('anomaly_heatmap.png')
    print("Anomaly Heatmap saved as 'anomaly_heatmap.png'")
    plt.close()
    
visualize_reconstruction_error(test_dataset, model, DEVICE)

# --- C. Confusion Matrix ---
# 1. Determine a robust threshold (using the 95th percentile of the NORMAL test subset's scores)
normal_scores_for_threshold = anomaly_scores[np.where(binary_labels == 0)[0]]
if len(normal_scores_for_threshold) > 0:
    # Use the 95th percentile of the normal test set's scores as the threshold.
    threshold = np.percentile(normal_scores_for_threshold, 95)
else:
    # Fallback threshold
    threshold = np.median(anomaly_scores) + np.std(anomaly_scores) * 2

# 2. Convert scores to predictions: 1=Anomaly, 0=Normal
predicted_labels = (anomaly_scores > threshold).astype(int) 

# 3. Compute Confusion Matrix
cm = confusion_matrix(binary_labels, predicted_labels)
cm_df = pd.DataFrame(cm, 
                     index=['True Normal (0)', 'True Anomaly (1)'], 
                     columns=['Pred Normal (0)', 'Pred Anomaly (1)'])

# 4. Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False, linewidths=.5, linecolor='black')
plt.title(f'Confusion Matrix (Threshold: {threshold:.4f})', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved as 'confusion_matrix.png'")
plt.close()
print("--- Visualization Complete ---")

# Re-run final print statements
print("\n--- Summary of Results ---")
print(f"Normal Class: {NORMAL_CLASS_NAME}")
print(f"Anomaly Detection AUC-ROC: {roc_auc:.4f}")
print("---")
print("First 5 Anomalies (High Scores):")
true_anomaly_indices = np.where(binary_labels == 1)[0]
for i in true_anomaly_indices[:5]:
    print(f"Score: {anomaly_scores[i]:<10.6f} | True Label: ANOMALY") 

print("\nFirst 5 Normal Samples (Low Scores):")
true_normal_indices = np.where(binary_labels == 0)[0]
for i in true_normal_indices[:5]:
    print(f"Score: {anomaly_scores[i]:<10.6f} | True Label: NORMAL")