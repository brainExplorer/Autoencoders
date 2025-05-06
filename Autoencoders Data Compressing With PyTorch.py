"""
Definition of problem : Data compression using autoencoders.
dataset : Fashion MNIST
model : Autoencoder
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# %% Loading dataset and preprocessing
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Batch size
batch_size = 128

# Train and test data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% autoencoder development

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),  # 28x28 = 784
            nn.ReLU(),
            nn.Linear(256, 64),  # 256
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),  # 64
            nn.ReLU(),
            nn.Linear(256, 28 * 28),  # 256
            nn.Sigmoid(),  # Output layer with sigmoid activation
            nn.Unflatten(1, (1, 28, 28))  # Reshape back to image dimensions
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %% callback : early stopping

class EarlyStopping:
    def __init__(self, patience=5, min_delta = 0.001 ):
        # how many epochs to wait before stopping
        self.patience = patience
        # minimum change to qualify as an improvement
        self.min_delta = min_delta
        # delta to consider as an improvement   
        self.best_loss = None
        # counter to count the number of epochs with no improvement
        self.counter = 0

    def __call__(self, val_loss):
        # if the best loss is None, set it to the current loss
        if self.best_loss is None:
            self.best_loss = val_loss
        # if the current loss is less than the best loss, set the best loss to the current loss and reset the counter
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        # if the current loss is greater than the best loss, increment the counter
        else:
            self.counter += 1

        # if the counter has reached the patience, return True to stop training
        if self.counter >= self.patience:
            return True
        return False


# %% training the model

# Hyperparameters
epochs = 50 # Number of epochs to train the model
learning_rate = 1e-3 # 1e-3 : 0.001

# Initialize the model, loss function, and optimizer
model = Autoencoder()   # Initialize the model
criterion = nn.MSELoss() # Loss function, using Mean Squared Error 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=5, min_delta=0.001) # Early stopping callback

# definition of training function
def train(model, train_loader, criterion, optimizer, early_stopping, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, inputs) # Compute loss between output and input
            loss.backward() # Backward pass 
            optimizer.step() # Update weights
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')    

        # Check for early stopping
        if early_stopping(avg_loss):
            print("Early stopping triggered {}.".format(epoch + 1))
            break

training = train(model, train_loader, criterion, optimizer, early_stopping, epochs)
        
        
# %% testing the model
# gaussian filter to smooth the images
from scipy.ndimage import gaussian_filter

def compute_ssim(img1, img2, sigma=1.5):
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute mean and variance
    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)

    mu1_sq = mu1 ** 2 # Square of mean of img1
    mu2_sq = mu2 ** 2 # Square of mean of img2
    mu1_mu2 = mu1 * mu2 
    
    sigma1_sq = gaussian_filter(img1 ** 2, sigma) - mu1_sq # Variance of img1
    sigma2_sq = gaussian_filter(img2 ** 2, sigma) - mu2_sq # Variance of img2

    # compute covariance
    sigma12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2 # Covariance of img1 and img2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return np.mean(ssim_map) # Return the mean SSIM score
    

def evaluate(model, test_loader, n_images=10):
    model.eval()    # Set the model to evaluation mode, which disables dropout and batch normalization
    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            outputs = model(inputs)
            break
    
    inputs = inputs.numpy()  # Convert to numpy array for visualization
    outputs = outputs.numpy()  # Convert to numpy array for visualization
    
    # Plotting the original and reconstructed images
    fig, axes = plt.subplots(2, n_images, figsize=(n_images, 3))
    ssim_scores = [] # List to store SSIM scores, which is a measure of similarity between two images
    for i in range(n_images):
        # Original images
        img1 = inputs[i].squeeze()  # Compress the channel dimension
        img2 = outputs[i].squeeze()  # Compress the recreated channel dimension
        
        ssim_score = compute_ssim(img1, img2) # Compute SSIM score
        ssim_scores.append(ssim_score) # Append the score to the list
        axes[0, i].imshow(img1, cmap='gray')
        axes[0, i].axis('off')
        axes[1,i].imshow(img2, cmap='gray')
        axes[1,i].axis('off')
    axes[0, 0].set_title('Original Images')
    axes[1, 0].set_title('Reconstructed Images')
    plt.show()
    
    avg_ssim = np.mean(ssim_scores) # Compute the average SSIM score
    print(f'Average SSIM score: {avg_ssim:.4f}') # Print the average SSIM score
    
    
evaluate(model, test_loader, n_images=10) # Evaluate the model on the test set    
    