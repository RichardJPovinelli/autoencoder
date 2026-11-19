import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)


# Define Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 28 * 28), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model, loss function, and optimizer
model = Autoencoder().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Forward pass
        output = model(img)
        loss = criterion(output, img)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Encode and decode some images from the test set
model.eval()
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.view(img.size(0), -1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        encoded_imgs = model.encoder(img).cpu().numpy()
        decoded_imgs = model(img).cpu().numpy()
        break

# Print reduced dimensions
print("Original dimension:", img.shape[1])  # 784
print("Reduced dimension:", encoded_imgs.shape[1])  # 16

# Visualize some encoded representations
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(encoded_imgs[i].reshape(8, 8), cmap="gray")  # Reshape for visualization
    plt.axis("off")
plt.suptitle("Encoded (Reduced) Representations")
plt.show()

# Visualize reconstructed images
plt.figure(figsize=(8, 4))
for i in range(10):
    # Original
    plt.subplot(2, 10, i + 1)
    plt.imshow(img[i].cpu().numpy().reshape(28, 28), cmap="gray")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.suptitle("Original (Top) and Reconstructed (Bottom) Images")
plt.show()
