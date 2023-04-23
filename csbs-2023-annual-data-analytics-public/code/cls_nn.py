# %%
# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV

# Read the data
data = pd.read_csv("../data/cleaned/04_08_finwlabel.csv", low_memory=False)

# %%
# Exclude the specified columns
exclude_columns = ['Reporting Period End Date', 'OCC Charter Number', 'OTS Docket Number',
                   'Primary ABA Routing Number', 'Financial Institution Filing Type', 'CERT', 'date',
                   'date_index', 'last', 'survival']

feature_columns = [col for col in data.columns if col not in exclude_columns]

# Select rows based on date_index for training and prediction
train_data = data[data['date_index'].isin([1, 6, 11, 16])].reset_index(drop=True)

# Drop columns with more than 10% NAs
columns_with_too_many_nas = train_data.columns[train_data.isna().mean() > 0.1]
train_data = train_data.drop(columns_with_too_many_nas, axis=1)

# Update feature_columns to only include columns that were not dropped
feature_columns = [col for col in feature_columns if col not in columns_with_too_many_nas]

# Fill remaining NAs with 0
train_data.fillna(0, inplace=True)

# Select a subset of train_data for feature selection
train_data_subset = train_data.sample(frac=0.005, random_state=42)

# Feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
cv = TimeSeriesSplit(n_splits=5)
rfecv = RFECV(RandomForestRegressor(random_state=42), step=1, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
rfecv.fit(train_data_subset[feature_columns], train_data_subset['last'])

# Select the top 15 features based on the feature importances
feature_importances = rfecv.estimator_.feature_importances_
sorted_features = sorted(zip(feature_columns, feature_importances), key=lambda x: x[1], reverse=True)
selected_features = [feature[0] for feature in sorted_features[:15]]
print(f'Selected features: {selected_features}')

# %%
# y_dict = dict(zip(data['CERT'],data['last']))
y = data[['CERT','last']]
idx = y.groupby(['CERT'])['last'].idxmax()
y = y.loc[idx]

# %%
def flatten_data(data, date_indices):
    flattened_data = []

    for _, group in data.groupby('CERT'):
        flattened_row = {'CERT': group['CERT'].iloc[0]}

        for index in date_indices:
            index_data = group[group['date_index'] == index]
            if not index_data.empty:
                for feature in selected_features:
                    flattened_row[f'{feature}_{index}'] = index_data[feature].values[0]

        flattened_data.append(flattened_row)

    return pd.DataFrame(flattened_data)


# Flatten the data and keep only the selected features
X = flatten_data(data[selected_features + ['CERT', 'date_index']], [1, 6, 11, 16])

# %%
from sklearn.model_selection import train_test_split

# Train/ Test split
all_data = X.merge(y)
all_data = all_data.fillna(0)

train_ratio = 0.8
test_ratio = 1 - train_ratio

# Split the DataFrame into train and test sets
train_data, test_data = train_test_split(all_data, test_size=test_ratio, random_state=42)

# %% [markdown]
# ### Model Eval

# %% [markdown]
# ## Try NN

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Assuming train_data and test_data are already defined
# Perform train/validation split
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(['CERT', 'last'], axis=1),
                                                  train_data[['last']],
                                                  test_size=0.2, random_state=42)

# Standardize the data (neural networks often perform better with standardized data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data.drop(['CERT', 'last'], axis=1))

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(test_data[['last']].values, dtype=torch.float32)


# %%
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create custom datasets
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# %%
import torch.nn as nn

class NeuralNetworkMultiClassification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetworkMultiClassification, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 20)
        self.layer7 = nn.Linear(20, num_classes)  # Change the output size to match the number of classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.relu(x)) + x
        x = self.drop(x)
        x = self.layer3(x)
        x = self.layer4(self.relu(x)) + x
        x = self.drop(x)
        x = self.relu(self.layer5(x))
        x = self.drop(x)
        x = self.relu(self.layer6(x))
        x = self.layer7(x)
        return nn.functional.softmax(x, dim=1)  # Apply softmax activation function to the output

input_size = X_train.shape[1]
num_classes = 20
model = NeuralNetworkMultiClassification(input_size, num_classes)
# Set device
device = torch.device("mps")
model.to(device)

# %%
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
# Training and validation loop
n_epochs = 200
for epoch in range(n_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    correct_preds = 0
    total_preds = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_batch -= 1
        y_batch = y_batch.squeeze()
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        train_loss += loss.item() * X_batch.size(0)
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(y_pred, 1)
        correct_preds += (predicted == y_batch).sum().item()
        total_preds += y_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracy = correct_preds / total_preds
    train_accuracies.append(train_accuracy)
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch -= 1
            y_batch = y_batch.squeeze()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(y_pred, 1)
            correct_preds += (predicted == y_batch).sum().item()
            total_preds += y_batch.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracy = correct_preds / total_preds
    val_accuracies.append(val_accuracy)
    # Print progress
    if (epoch % 10 == 0):
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


