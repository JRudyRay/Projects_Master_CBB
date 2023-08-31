# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None

    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles",
                                                                                                  axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test


class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data
        # and then used to extract features from the training and test data.
        #self.fc1 = nn.Identity()
        #for params in self.fc1.parameters():
        #    params.requires_grad = False
        self.fc1 = nn.Sequential(
            nn.Linear(1000,600),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(600,300),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(300,1)
        )

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture
        # defined in the constructor.
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def train_model(model: nn.Module, tr_loader: DataLoader, epoch, n_epochs, criterion, optimizer):
    loop = tqdm(tr_loader)
    model.train()
    epoch_train_loss = 0.0

    for idx, (x_tr, y_tr) in enumerate(loop):
        x_tr = x_tr.to(device)
        y_tr = y_tr.to(device)

        # Set optimizer gradients to zero
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_tr)
        y_pred = torch.squeeze(y_pred)
        loss = criterion(y_pred, y_tr)
        epoch_train_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Progress bar
        loop.set_description(f'Epoch {epoch + 1}/{n_epochs} Training')
        loop.set_postfix(loss=loss.item())

    epoch_train_loss = epoch_train_loss / len(tr_loader)

    return epoch_train_loss


def validate_model(model: nn.Module, val_loader: DataLoader, epoch, n_epochs, criterion):
    loop = tqdm(val_loader)
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for idx, (x_val, y_val) in enumerate(loop):
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # Evaluate the model
            y_pred = model(x_val)
            y_pred = torch.squeeze(y_pred)
            loss = criterion(y_pred, y_val)
            epoch_val_loss += loss.item()

            # Progress bar
            loop.set_description(f'Epoch {epoch + 1}/{n_epochs} Validation')
            loop.set_postfix(loss=loss.item())

    epoch_val_loss = epoch_val_loss / len(val_loader)

    return epoch_val_loss


def plot_loss(epochs, train_losses, val_losses):
    plt.yscale('log')
    plt.plot(epochs + 1, train_losses, c='b', label='training')
    plt.plot(epochs + 1, val_losses, c='r', label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Model evaluation')
    plt.savefig('model_eval1.png')


def make_feature_extractor(x, y, batch_size=256, eval_size=1000, n_epochs=8):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
            y: np.ndarray, the labels of the pretraining set
              batch_size: int, the batch size used for training
              eval_size: int, the size of the validation set

    output: make_features: function, a function which can be used to extract features from the training and test data
      """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    tr_dataset = TensorDataset(x_tr, y_tr)
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size,
                           pin_memory=True, shuffle=True, num_workers=2)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            pin_memory=True, num_workers=2)

    # model declaration
    model = Net()
    model.to(device)

    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set
    # to monitor the loss.
    epochs = np.arange(n_epochs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = np.zeros(n_epochs)
    val_losses = np.zeros(n_epochs)

    for epoch in epochs:
        train_loss = train_model(model=model, tr_loader=tr_loader,
                                 epoch=epoch, n_epochs=n_epochs,
                                 criterion=criterion, optimizer=optimizer)

        val_loss = validate_model(model=model, val_loader=val_loader,
                                  epoch=epoch, n_epochs=n_epochs,
                                  criterion=criterion)

        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

    plot_loss(epochs, train_losses, val_losses)
    print(f'Training\tLosses: {train_losses}')
    print(f'Validation\tLosses: {val_losses}')

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        x = torch.tensor(x, dtype=torch.float) # to tensor for GPU computation
        x = x.to(device)

        # simple forward hook to extract the output of layer fc1
        features = {}

        def hook(m, input, output): # dictionary for the layers
            features['features'] = output.detach().cpu().numpy() # saving extracted features

        with torch.no_grad():
            model.fc1.register_forward_hook(hook) # this is where the transfer learning is make
            _ = model(x)
            x = features['features'] # overwriting x with the features
            pd.DataFrame(x)

        return x

    return make_features


def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline

    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """

        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new

    return PretrainedFeatures


def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.

    #model = LinearRegression()
    model = RidgeCV(alphas=np.logspace(-3, 4, 2000), cv=5) # internal crossvalidation with ridge
    #model = LassoCV(cv = 8)
    #model = SGDRegressor()
    #model = SVR()

    return model


# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()

    #x_pretrain, x_preval, y_pretrain, y_preval = train_test_split(x_pretrain, y_pretrain, random_state=0, test_size=0.25)

    # x_test should be np.ndarray, store index and use it later to save the predictions
    index = x_test.index
    x_test = x_test.to_numpy()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy
    # features from available initial features
    n_epochs = 20
    feature_extractor = make_feature_extractor(x_pretrain, y_pretrain, n_epochs=n_epochs, eval_size=15000)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})

    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    pipeline = Pipeline(steps=[('extractor', PretrainedFeatureClass(feature_extractor='pretrain')),
                                  ('scaler', StandardScaler()),
                                  ('regressor', regression_model)], verbose=True)

    """
    Use the pipeline to check, if the pipeline works for the pretrain data
    #pipeline.fit(x_pretrain, y_pretrain)
    #y_prepredict = pipeline.predict(x_preval)
    #error = mean_squared_error(y_prepredict, y_preval)
    #print(f"x_pretrain score:{pipeline.score(x_pretrain,y_pretrain)}")
    #print(f"Pretraining error: {error}")
    """

    # use the pipeline on the training data with validation
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.3)
    pipeline.fit(x_train, y_train)
    #y_predict = pipeline.predict(x_val)

    #error = mean_squared_error(y_val, y_predict, squared=False)
    print(pipeline.score(x_train, y_train))
    #print(error)


    y_pred = pipeline.predict(x_test)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")