import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from torchvision.models.resnet import ResNet50_Weights
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # Define a transform to pre-process the images
    train_transforms = transforms.Compose([
        transforms.Resize(size = 232),
        transforms.CenterCrop(size = 224), 
        transforms.ToTensor(), #transforms to tensor and rescales [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=12
    ) # my laptop can handle num_workers=12

    # Define a model for extraction of the embeddings
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential()

    embedding_size = 2048 # resnet has 2048 features
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    # Use the model to extract the embeddings
    with torch.no_grad(): # saves memory and increases speed
        model.eval() # put the model in evaluation mode
        for i, (inputs, _) in enumerate(train_loader):
            outputs = model(inputs) # passing the inputs through the NN
            outputs = outputs.view(outputs.size(0), -1) # flatten the output
            start_index = i * 64 # start and end indices of the batch
            end_index = start_index + outputs.size(0)
            embeddings[start_index:end_index, :] = outputs.cpu().numpy()
            torch.cuda.empty_cache() # clear the GPU memory

    # Save the embeddings
    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    # Get the triplets from the file
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line.strip()) # Need to .strip() here

    # Load the image embeddings
    embeddings = np.load('dataset/embeddings.npy')

    # Normalize the embeddings
    embeddings = normalize(embeddings, axis=1)

    # Map filenames to embeddings
    train_dataset = datasets.ImageFolder(root="dataset/", transform=None)
    filenames = [os.path.splitext(os.path.basename(s[0]))[0] for s in train_dataset.samples]
    file_to_embedding = {}
    for i, filename in enumerate(filenames):
        file_to_embedding[filename] = embeddings[i]

    X = []
    y = []
    # Use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        filenames = t.split() # split up the triplets
        if all(f in file_to_embedding for f in filenames): # checking if all the filenames are in the file_to_embedding
            embeddings_triplet = [file_to_embedding[f] for f in filenames]  
            X.append(np.concatenate(embeddings_triplet))
            y.append(1)
            # Generating negative samples (data augmentation), for training purposes
            if train:
                X.append(np.concatenate([embeddings_triplet[0], embeddings_triplet[2], embeddings_triplet[1]]))
                y.append(0)

    X = np.vstack(X)
    y = np.vstack(y).astype(np.float32) # due to failure in train_model need to convert it here
   
    return X, y


def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 8):
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                        pin_memory=True, num_workers=num_workers)
    return loader


class Net(nn.Module):
    """
    The alternative model class, which defines our classifier.
    """
    def __init__(self, input_size):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc5(x)
        x = torch.sigmoid(x)
        return x.view(-1, 1)



def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net(input_size = 6144)  # need to pass the right input size to the model
    model.to(device)
    model.train()

    n_epochs = 10

    criterion = nn.BCEWithLogitsLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters())  # Adam optimizer

    epoch_losses = []  

    for epoch in range(n_epochs): 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero out the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.view(inputs.size(0), -1))  # reshaping the input tensor to equal the size of the input to fc layer
            outputs = outputs.float()  # cast output tensor to float type
            loss = criterion(outputs, labels.float())  # passing the float type labels tensor to the loss function

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
 
            # Sum up the loss
            running_loss += loss.item()

        # Compute the average loss for the epoch
        avg_loss = running_loss / (i + 1)
        epoch_losses.append(avg_loss)  # appending the average loss
        print('Epoch [%d] loss: %.3f' % (epoch + 1, avg_loss)) # print epoch and loss during training

    # plot the losses
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        print("Generating embeddings...")
        generate_embeddings()
        print("Embeddings generated")

    # load the training and testing data
    print("Loading data...")
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)
    print("Data loaded")

    # Create data loaders for the training and testing data
    print("Creating data loaders...")
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)
    print("Data loaders created")

    # define a model and train it
    print("Training model...")
    model = train_model(train_loader)
    print("Finished training")

    # test the model on the test data
    print("Testing model...")
    test_model(model, test_loader)
    print("Finished testing")
    print("Results saved to results.txt")

