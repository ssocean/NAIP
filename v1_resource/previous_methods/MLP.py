import string
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

 
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        latent = self.linear(x)
        return self.linear2(latent)
        # return nn.Sigmoid()(self.linear(x))

 
class CitationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

 
def preprocess_and_scale_data(df):
    df['title_length'] = df['title'].apply(lambda x: len(x.split()))   
    df['title_punctuation'] = df['title'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))   
    df['pdf_page_num'] = df['pdf_page_num'].fillna(df['pdf_page_num'].mean())   
    df['SMP'] = df['SMP'].fillna(df['SMP'].mean())   
    features = df[['pdf_page_num', 'title_length', 'title_punctuation', 'Ref_num', 'SMP']].values
    labels = df['cites'].values

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)   
    return scaled_features, labels

 
train_data = pd.read_csv('NAID_train_extrainfo.csv')
train_scaled_features, train_targets = preprocess_and_scale_data(train_data)
test_data = pd.read_csv('NAID_test_extrainfo.csv')
test_scaled_features, test_targets = preprocess_and_scale_data(test_data)

 
train_dataset = CitationDataset(train_scaled_features, train_targets)
test_dataset = CitationDataset(test_scaled_features, test_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearRegressionModel(train_scaled_features.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

 
model.train()
for epoch in range(50):
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

 
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = model(features)
        predictions.extend(outputs.cpu().numpy().flatten().tolist())
        actuals.extend(labels.numpy().tolist())

 
mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
print(f'Test MSE: {mse}')

 
ndcg_value = ndcg_score([actuals], [predictions], k=20)   
print(f'NDCG Score: {ndcg_value}')
