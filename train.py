from interface import model_params, training_params, wandb_params
from model import compare_sequences
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from model import test_model_instance, seq2seq_attn, seq2seq, train, test
import torch
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = training_params["batch_size"]

x_train = np.loadtxt("akshar_sequences//x_train.csv", delimiter=",", dtype=int)
y_train = np.loadtxt("akshar_sequences//y_train.csv", delimiter=",", dtype=int)
x_test = np.loadtxt("akshar_sequences//x_test.csv", delimiter=",", dtype=int)
y_test = np.loadtxt("akshar_sequences//y_test.csv", delimiter=",", dtype=int)
x_val = np.loadtxt("akshar_sequences//x_val.csv", delimiter=",", dtype=int)
y_val = np.loadtxt("akshar_sequences//y_val.csv", delimiter=",", dtype=int)

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index]).long() 
        y = torch.from_numpy(self.y[index]).long() 
        return x, y
    
    def __len__(self):
        return len(self.x)

train_dataset = SequenceDataset(x_train, y_train)
val_dataset = SequenceDataset(x_val, y_val)
test_dataset = SequenceDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

VOCAB_SIZE = model_params["vocab_size"]
EMBEDDING_DIM = model_params["embedding_size"]
HIDDEN_DIM = model_params["hidden_size"]
EPOCHS = training_params["epochs"]
NUM_LAYERS_ENCODER = model_params["num_layers_encoder"]
NUM_LAYERS_DECODER = model_params["num_layers_decoder"]
DROPOUT = model_params["dropout"]
BIDIRECTIONAL = model_params["bidirectional"]
CELL_TYPE_ENCODER = model_params["encoder_cell"]
CELL_TYPE_DECODER = model_params["decoder_cell"]
TEACHER_FORCING = model_params["teacher_forcing"]
MAX_SEQ_SIZE = model_params["max_seq_size"]

model = train(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER, DROPOUT, BIDIRECTIONAL, 
              CELL_TYPE_ENCODER, CELL_TYPE_DECODER, TEACHER_FORCING, BATCH_SIZE, MAX_SEQ_SIZE, EPOCHS, model_params["attention"], 
              train_dataset, val_dataset)



