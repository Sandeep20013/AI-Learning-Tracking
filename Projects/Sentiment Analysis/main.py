import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim

from helpers.dataset import IMDBDataset
from helpers.train_model import train_model
from sklearn.model_selection import train_test_split
from models.sentiment_rnn import SentimentRNN
from models.sentiment_lstm import SentimentLSTM
from helpers.early_stopping import EarlyStopping
from helpers.eval_model import evaluate_model
from timeit import default_timer as timer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('data/IMDB Dataset.csv')
df['sentiment'] = (df['sentiment'].str.lower() == 'positive').astype(int)
print(df.head())

train_df, test_df = train_test_split(df, test_size = 0.2, stratify = df['sentiment'], random_state = 42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['sentiment'], random_state=42)

# Create train_dataset
train_dataset = IMDBDataset(train_df, build_vocab=True, max_len=250)
val_dataset = IMDBDataset(val_df, build_vocab=False, vocab=train_dataset.vocab, max_len=250)
test_dataset = IMDBDataset(test_df, build_vocab=False, vocab=train_dataset.vocab, max_len=250)

# Use dataloader to load data into model
torch.manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

model_v1 = SentimentRNN(
    vocab_size= len(train_dataset.vocab),
    embed_dim = 64, 
    hidden_dim= 128,
    output_dim=1
).to(device)
optimizer_v1 = torch.optim.Adam(model_v1.parameters(), lr=1e-3)
loss_fn_v1 = nn.BCEWithLogitsLoss()
early_stop = EarlyStopping(patience=5, min_delta=0.01)

epochs = 10
# torch.manual_seed(42)
# trained_model_rnn = train_model(model = model_v1,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             optimizer=optimizer_v1,
#             loss_fn=loss_fn_v1,
#             device=device,
#             early_stopping=early_stop)
# eval_metrics_1 = evaluate_model(trained_model_rnn, test_loader, device)
model_v2 = SentimentLSTM(
    vocab_size= len(train_dataset.vocab),
    embed_dim = 64, 
    hidden_dim= 128,
    output_dim=1
).to(device)
optimizer_v2 = torch.optim.Adam(model_v2.parameters(), lr=1e-3)
loss_fn_v2 = nn.BCEWithLogitsLoss()
early_stop = EarlyStopping(patience=5, min_delta=0.01)
trained_model_lstm= train_model(model = model_v2,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer_v2,
            loss_fn=loss_fn_v2,
            device=device,
            early_stopping=early_stop)
eval_metrics_1 = evaluate_model(trained_model_lstm, test_loader, device)

