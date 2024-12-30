from transformer import Transformer
from text_dataset import TextDataset

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from datasets import load_dataset
from transformers import AutoTokenizer 

from sklearn.model_selection import train_test_split

def collate_fn(batch):
    input_seqs, target_seqs = zip(*batch)

    input_seqs = pad_sequence(input_seqs, padding_value=tokenizer.pad_token_id, batch_first=True)
    target_seqs = pad_sequence(target_seqs, padding_value=tokenizer.pad_token_id, batch_first=True)
    return input_seqs, target_seqs

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_texts = dataset["train"]["text"]
train_texts, valid_texts = train_test_split(train_texts, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_texts, tokenizer)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

model_params = {
    'batch_size' : 32,
    'vocab_size' : tokenizer.vocab_size,
    'model_dim' : 256,
    'num_heads' : 4,
    'num_layers' : 2,
    'ff_dim' : 512,
    'max_len' : 512,
    'num_epochs' : 10,
    'lr' : 0.001
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Transformer(
    model_params['vocab_size'], 
    model_params['num_layers'], 
    model_params['num_heads'],
    model_params['model_dim'], 
    model_params['ff_dim'], 
    model_params['max_len']).to(device)

criterion = CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr = model_params['lr'])

for epoch in range(model_params['num_epochs']):
    model.train()
    total_loss = 0.0

    for input_seq, target_seq in train_loader:

        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        optimizer.zero_grad()

        output = model(input_seq, target_seq[:,:-1])

        loss = criterion(output.view(-1, model_params['vocab_size']), target_seq[:, 1:].contiguous().view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{model_params['num_epochs']}], Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "MyTransformer_model.pth")