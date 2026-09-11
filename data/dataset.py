import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from config import DIGIT


class MNISTDigitDataset(Dataset):

    def __init__(self, digit: int = DIGIT, train: bool = True, root: str = "./data/raw"):

        transform = transforms.Compose([
            #image to Tensor
            transforms.ToTensor(),
            #rescale to [-1,1]
            transforms.Normalize((0.5,), (0.5,)),
        ])

        full_dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform) #28x28 pixel with values in {0,...,255} (0=black, 1=white)

        #Get indices for digit == 0 (i.e. only "good" examples)
        mask = full_dataset.targets == digit
        indices = torch.where(mask)[0]

        #Filter according to indices and get images (1,28,28) (stacked: (n,1,28,28))
        self.data = torch.stack([full_dataset[i][0] for i in indices])

        #(n,1,28,28)-->(n,784), i.e. flattened to R^d, d=784=28*28
        self.data = self.data.view(self.data.size(0), -1)

    def __len__(self):
        return self.data.size(0) #number of samples

    def __getitem__(self, idx):
        return self.data[idx] #returns X_idx in R^d (shape (784,))


if __name__ == "__main__":

    ds = MNISTDigitDataset(digit=0, train=True)
    print(f"Number of training data points (for digit 0): {len(ds)}")
    print(f"Sample shape: {ds[0].shape}")
    print(f"Range: [{ds[0].min():.2f}, {ds[0].max():.2f}]")

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch shape (size 8): {batch.shape}")