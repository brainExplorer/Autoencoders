# Fashion MNIST Autoencoder for Data Compression

This project demonstrates data compression using an Autoencoder trained on the Fashion MNIST dataset. It includes training, early stopping, and evaluation using SSIM (Structural Similarity Index Measure).

---

## 🧠 Model Architecture: Autoencoder

An Autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data. The encoder compresses the input, and the decoder reconstructs it.

### Encoder:
- Input: 28x28 image (flattened to 784)
- Fully connected layer → 256 neurons → ReLU
- Fully connected layer → 64 neurons → ReLU

### Decoder:
- Fully connected layer → 256 neurons → ReLU
- Fully connected layer → 784 neurons → Sigmoid
- Reshape to (1, 28, 28)

---

## 📦 Dataset

- **Fashion MNIST** from `torchvision.datasets`
- Contains 60,000 training images and 10,000 test images of 28x28 grayscale clothing items.
- Downloaded automatically if not present.

---

## 🔄 Training

Training includes:
- Mean Squared Error (MSE) loss
- Adam optimizer (`lr = 0.001`)
- Batch size: 128
- Early stopping with:
  - `patience = 5`
  - `min_delta = 0.001`

```python
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
early_stopping = EarlyStopping(patience=5, min_delta=0.001)
train(model, train_loader, criterion, optimizer, early_stopping, epochs=50)
```

---

## 🛑 EarlyStopping Callback

A custom early stopping class monitors validation loss and stops training when improvement falls below a threshold over a given number of epochs.

```python
class EarlyStopping:
    ...
```

---

## 🔍 Evaluation

Reconstructed images are compared to the originals using the Structural Similarity Index (SSIM), computed via a Gaussian filter.

```python
def compute_ssim(img1, img2, sigma=1.5):
    ...
```

### Visualization:
Displays 10 original and 10 reconstructed images using `matplotlib`.

```python
evaluate(model, test_loader, n_images=10)
```

---

## 📊 Results

- **Loss** decreases consistently until early stopping is triggered.
- **SSIM** is calculated between original and reconstructed images to evaluate visual similarity.

---

## 🧰 Dependencies

- torch
- torchvision
- matplotlib
- numpy
- scipy

Install with:

```bash
pip install torch torchvision matplotlib numpy scipy
```

---

## 📁 Project Structure

```
.
├── autoencoder_fashion_mnist.py
├── README.md
└── data/
    └── FashionMNIST/
```

---

## 📸 Sample Output

| Original | Reconstructed |
|----------|---------------|
| _Insert image examples here_ |

_(Add saved sample image comparisons for visual inspection)_

---

## 📌 Notes

- Consider tuning the encoder's bottleneck (e.g., 32 or 16 latent features) for higher compression.
- You can switch to convolutional autoencoders for better spatial feature extraction.

---

## 👨‍💻 Author

- Created by Ümit YAVUZ
- GitHub: brainExplorer
