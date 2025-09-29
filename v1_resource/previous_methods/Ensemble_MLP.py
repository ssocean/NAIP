import string
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class CitationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def preprocess_and_scale_data(df, arxiv_type):
    df = df[df['arxiv_type'] == arxiv_type]
    df['title_length'] = df['title'].apply(lambda x: len(x.split()))
    df['title_punctuation'] = df['title'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))
    df['pdf_page_num'] = df['pdf_page_num'].fillna(df['pdf_page_num'].mean())
    df['SMP'] = df['SMP'].fillna(df['SMP'].mean())
    features = df[['pdf_page_num', 'title_length', 'title_punctuation', 'Ref_num', 'SMP']].values
    labels = df['cites'].values

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels

data = pd.read_csv('NAID_train_extrainfo.csv')
arxiv_types = data['arxiv_type'].unique()

models = {}
train_loaders = {}
test_loaders = {}
for arxiv_type in arxiv_types:
    scaled_features, targets = preprocess_and_scale_data(data, arxiv_type)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, targets, test_size=0.2, random_state=42)
    train_dataset = CitationDataset(X_train, y_train)
    test_dataset = CitationDataset(X_test, y_test)
    train_loaders[arxiv_type] = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loaders[arxiv_type] = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = LinearRegressionModel(X_train.shape[1]).to(device)
    models[arxiv_type] = model

 
for arxiv_type in arxiv_types:
    model = models[arxiv_type]
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(50):
        for features, labels in train_loaders[arxiv_type]:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()} for arxiv_type {arxiv_type}')

 
predictions = []
actuals = []
for arxiv_type in arxiv_types:
    model = models[arxiv_type].eval()
    with torch.no_grad():
        for features, labels in test_loaders[arxiv_type]:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
            actuals.extend(labels.numpy().tolist())

mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
print(f'Test MSE: {mse}')

ndcg_value = ndcg_score([actuals], [predictions], k=20)
print(f'NDCG Score: {ndcg_value}')
