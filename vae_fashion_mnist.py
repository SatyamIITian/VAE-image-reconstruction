import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from uuid import uuid4

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Load Fashion-MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Class labels for Fashion-MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Colors for class visualization
class_colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Initialize and Train VAE
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training function (simplified for pretrained effect)
def train_vae(epochs=10):
    vae.train()
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset):.4f}')

# Evaluate and Reconstruct
def evaluate_vae():
    vae.eval()
    mse_loss = nn.MSELoss(reduction='mean')
    class_mse = {i: [] for i in range(10)}
    all_reconstructions = []
    all_originals = []
    all_labels = []
    all_latents = []
    num_samples_per_class = 2  # Reduced to 2 samples per class for larger images
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            recon, mu, _ = vae(data)
            mse = mse_loss(recon, data.view(-1, 784)).item()
            
            for i, label in enumerate(labels):
                class_mse[label.item()].append(mse)
                if len([l for l in all_labels if l == label.item()]) < num_samples_per_class:
                    all_reconstructions.append(recon[i].view(28, 28).cpu().numpy())
                    all_originals.append(data[i].view(28, 28).cpu().numpy())
                    all_labels.append(label.item())
                    all_latents.append(mu[i].cpu().numpy())
    
    # Compute average MSE per class
    class_avg_mse = {class_names[i]: np.mean(class_mse[i]) for i in range(10)}
    
    return class_avg_mse, all_originals, all_reconstructions, all_labels, all_latents

# Visualization Functions
def visualize_reconstructions(originals, reconstructions, labels, num_samples=2):
    total_images = len(originals)  # Should be 20 (2 samples x 10 classes)
    plt.figure(figsize=(8, 3 * total_images), dpi=300)  # Increased size for larger images
    
    for i in range(total_images):
        label = labels[i]
        # Original image (grayscale)
        plt.subplot(total_images, 2, i * 2 + 1)
        plt.imshow(originals[i], cmap='gray')
        plt.gca().set_axis_off()
        plt.gca().add_patch(plt.Rectangle((0, 0), 28, 28, linewidth=2, edgecolor=class_colors[label], fill=False))
        if i == 0:
            plt.title("Original", fontsize=12)
        plt.text(14, -5, class_names[label], ha='center', fontsize=10, color=class_colors[label])
        
        # Reconstructed image (grayscale)
        plt.subplot(total_images, 2, i * 2 + 2)
        plt.imshow(reconstructions[i], cmap='gray')
        plt.gca().set_axis_off()
        plt.gca().add_patch(plt.Rectangle((0, 0), 28, 28, linewidth=2, edgecolor=class_colors[label], fill=False))
        if i == 0:
            plt.title("Reconstructed", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('reconstructions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_latent_space(latents, labels):
    latents = np.array(latents)
    if latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
    else:
        latents_2d = latents
    
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(10):
        mask = np.array(labels) == i
        plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], c=[class_colors[i]], label=class_names[i], alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.tight_layout()
    plt.savefig('latent_space.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_mse_bar(class_avg_mse):
    df = pd.DataFrame(list(class_avg_mse.items()), columns=['Class', 'Average MSE'])
    df = df.sort_values('Average MSE')
    
    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(df['Class'], df['Average MSE'], color='skyblue', edgecolor='black')
    
    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    plt.ylim(0, 0.02)  # Fixed y-axis range for context
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel('Average MSE', fontsize=12)
    plt.title('Class-wise Reconstruction MSE\n(Lower MSE = Better Reconstruction)', fontsize=14)
    plt.tight_layout()
    plt.savefig('mse_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Summarize Results
def summarize_results(class_avg_mse):
    df = pd.DataFrame(list(class_avg_mse.items()), columns=['Class', 'Average MSE'])
    df = df.sort_values('Average MSE')
    print("\nClass-wise Average MSE:")
    print(df)
    
    # Save to CSV
    df.to_csv('class_mse_summary.csv', index=False)
    
    # Interpretation
    print("\nInterpretation:")
    print("Classes with lower MSE (e.g., Trousers, Bags) are likely reconstructed more accurately due to simpler shapes and less variability.")
    print("Classes with higher MSE (e.g., Shirts, Coats) may have more complex patterns or overlap with other classes, making reconstruction harder.")

# Main Execution
if __name__ == "__main__":
    print("Training VAE...")
    train_vae(epochs=10)
    
    print("Evaluating VAE...")
    class_avg_mse, originals, reconstructions, labels, latents = evaluate_vae()
    
    print("Visualizing reconstructions...")
    visualize_reconstructions(originals, reconstructions, labels)
    visualize_latent_space(latents, labels)
    visualize_mse_bar(class_avg_mse)
    
    print("Summarizing results...")
    summarize_results(class_avg_mse)