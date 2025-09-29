 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import nltk
from tqdm import tqdm

nltk.download('punkt')
nltk.download('omw-1.4')

# def dcg_at_k(scores, k):
#     """
#     scores: a list of relevance scores in predicted order
#     k: number of results to consider
#     """
#     scores = np.asfarray(scores)[:k]
#     return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
#
# def ndcg_at_k(predicted_scores, true_scores, k):
#     """
#     predicted_scores: model's predicted scores
#     true_scores: ground truth scores
#     k: number of results to consider
#     """
#     idcg = dcg_at_k(sorted(true_scores, reverse=True), k)
#     dcg = dcg_at_k(predicted_scores, k)
#     return dcg / idcg if idcg > 0 else 0

import torch
import numpy as np

from sklearn.metrics import ndcg_score

def NDCG_k(predictions, labels, k=20):
    print(print(predictions.shape, labels.shape))
    predictions = predictions.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()
    if len(predictions) < k:
        return -1

     
    ndcg = ndcg_score([labels], [predictions], k=k)

    print("Average NDCG:", ndcg)
    return ndcg
 
# nltk.download('stopwords')
# nltk.download('wordnet')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

     
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return words


 
class PapersDataset(Dataset):
    def __init__(self, dataframe, doc2vec_model,target_type='TNCSI'):
        """
        dataframe: 传入的DataFrame，包含训练或测试数据
        doc2vec_model: 已训练的Doc2Vec模型
        """
        self.dataframe = dataframe
        self.doc2vec_model = doc2vec_model
        self.target_type = target_type
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        metadata = f"{row['title']} {row['abstract']}"
        processed_text = preprocess_text(metadata)
        vector = self.doc2vec_model.infer_vector(processed_text)
        if self.target_type.startswith('TNCSI'):
            label = row[self.target_type]
        else:
            label = row['cites']
        return torch.tensor(vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

 
def train_doc2vec(documents):
    tagged_data = [TaggedDocument(words=preprocess_text(doc), tags=[i]) for i, doc in enumerate(documents)]
    model = Doc2Vec(tagged_data, vector_size=100, window=2, min_count=1, workers=4, epochs=40)
    return model


 
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
         
        weights = torch.tanh(self.linear(lstm_output))
        weights = torch.softmax(weights, dim=1)
         
        weighted = torch.mul(lstm_output, weights.expand_as(lstm_output))
         
        return torch.sum(weighted, dim=1)

class CitationModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,target_type='TNCSI'):
        super(CitationModel, self).__init__()
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)   
        self.fc = nn.Linear(hidden_dim * 2, 1)   
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.target_type = target_type

    def forward(self, x):
        lstm_out, _ = self.bi_lstm(x)
        attention_out = self.attention(lstm_out)
        output = self.fc(attention_out)
        if self.target_type.startswith('TNCSI'):
            output = self.sigmoid(output)
        else:
            output = self.relu(output)
        return output



def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    loss_history = []   
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for inputs,targets in train_loader:
            inputs, targets = inputs.to(device).unsqueeze(1), targets.to(device).unsqueeze(1)   
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
        # else:
        #     for inputs,_,targets in train_loader:
         
        #         optimizer.zero_grad()
        #         outputs = model(inputs)
        #
        #         loss = criterion(outputs, targets)
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()
        #     avg_loss = total_loss / len(train_loader)
        #     loss_history.append(avg_loss)
        #     print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
     
    print("Training complete. Loss history:")
    print(loss_history)

# Evaluation function for NDCG
def evaluate_model(model, test_loader, device,k=20):
    model.eval()
    pred_scores = []
    target_scores = []
    with torch.no_grad():
        total_loss = 0


        for inputs, target in test_loader:
            inputs = inputs.to(device).unsqueeze(1)  # Ensure input is correctly shaped
            outputs = model(inputs)

            # Flatten outputs and targets for NDCG computation
            predicted_scores = outputs.squeeze()
            true_scores = target.squeeze()
            loss = nn.MSELoss()(predicted_scores.detach().cpu(),true_scores.detach().cpu())
            total_loss += loss.item()
            print(predicted_scores)
            print(true_scores)
            print('-'*50)
            pred_scores.append(outputs)
            target_scores.append(true_scores)

    avg_loss = total_loss / len(test_loader)
    print(f'AVG MSE:{avg_loss}')

    all_pred = torch.cat(pred_scores, dim=0).squeeze()
    all_GT = torch.cat(target_scores, dim=0).squeeze()

    # all_pred = torch.Tensor(pred_scores)
    # all_GT = torch.Tensor(target_scores)
    ndcg = NDCG_k(all_pred, all_GT,k=k)
    print(ndcg)

    return ndcg
# Main function
def main():
    csv_file = r' Desktop\NAID_train_extrainfo.csv'
    target_type = 'TNCSI_SP'

    train_data = pd.read_csv(csv_file)
    test_data = pd.read_csv(r' Desktop\NAID_test_extrainfo.csv')

    # Train the Doc2Vec model on training data abstracts
    train_documents = train_data['abstract'].tolist()
    doc2vec_model = train_doc2vec(train_documents)

    # Create training and testing datasets

    train_dataset = PapersDataset(dataframe=train_data, doc2vec_model=doc2vec_model,target_type=target_type)
    test_dataset = PapersDataset(dataframe=test_data, doc2vec_model=doc2vec_model,target_type=target_type)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CitationModel(embedding_dim=100, hidden_dim=1024,target_type=target_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    ep = 30

    train_model(model, train_loader, criterion, optimizer, epochs=ep, device=device)
    torch.save(model.state_dict(), f'LSTM-{target_type}-{ep}.pth')
     
    # model.load_state_dict(torch.load(f'LSTM-{target_type}-{ep}.pth'))

    # Evaluate using NDCG
    evaluate_model(model, test_loader,  device=device)

if __name__ == '__main__':
    main()