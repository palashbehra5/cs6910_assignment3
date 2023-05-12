import torch
import torch.nn as nn
import random
import itertools
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class seq2seq(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers_encoder, num_layers_decoder, 
                 dropout, bidirectional, encoder_cell_type, decoder_cell_type, teacher_forcing, 
                 batch_size, max_seq_size, debugging = False):

        super(seq2seq, self).__init__()

        self.output_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.encoder_cell_type = encoder_cell_type
        self.decoder_cell_type = decoder_cell_type
        self.teacher_forcing_prob = teacher_forcing
        self.debugging = debugging
        self.batch_size = batch_size
        self.max_seq_size = max_seq_size

        self.dropout = nn.Dropout(self.dropout_prob)
        self.embedding_encoder = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.embedding_dim).to(device)
        self.embedding_decoder = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.embedding_dim).to(device)

        self.rnn_encoder = self.cell(num_layers_encoder, encoder_cell_type, bool(self.bidirectional))
        self.rnn_decoder = self.cell(num_layers_decoder, decoder_cell_type, 0)

        # Final layer for calculating probabilities
        self.fc1 = nn.Linear(self.hidden_dim, self.output_size)

    def cell(self, num_layers, cell_type, bidirectional):

        # Defining Cells
        cells = {

            "LSTM" : nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, 
                            dropout=self.dropout_prob, bidirectional=bool(bidirectional)).to(device),

            "GRU" : nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, 
                            dropout=self.dropout_prob, bidirectional=bool(bidirectional)).to(device),

            "RNN" : nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, 
                            dropout=self.dropout_prob, bidirectional=bool(bidirectional)).to(device)

        }

        return cells[cell_type]
    

    def initialize_decoder_state(self, encoder_cell_type, encoder_state, encoder_cell, decoder_cell_type, bidirectional,
                             num_decoder_layers, num_encoder_layers):

        batch_size = encoder_state[0].size(0)  # Get the batch size from the encoder states

        if encoder_cell_type == "LSTM":
            if bidirectional:
                forward_state, backward_state = encoder_state[:num_encoder_layers], encoder_state[num_encoder_layers:]
                forward_cell, backward_cell = encoder_cell[:num_encoder_layers], encoder_cell[num_encoder_layers:]
                encoder_state = (torch.mean(forward_state, dim=0) + torch.mean(backward_state, dim=0)) / 2
                encoder_cell = (torch.mean(forward_cell, dim=0) + torch.mean(backward_cell, dim=0)) / 2
            else:
                encoder_state = torch.mean(encoder_state, dim=0)
                encoder_cell = torch.mean(encoder_cell, dim=0) if decoder_cell_type == "LSTM" else None
        else:
            forward_state = encoder_state[:num_encoder_layers]
            backward_state = encoder_state[num_encoder_layers:] if bidirectional else None
            encoder_state = (torch.mean(forward_state, dim=0) + torch.mean(backward_state, dim=0)) / 2 if bidirectional else torch.mean(forward_state, dim=0)
            encoder_cell = None

        decoder_state = encoder_state.unsqueeze(0).expand(num_decoder_layers, batch_size, -1)

        if decoder_cell_type == "LSTM":
            decoder_cell = decoder_state
        else:
            decoder_cell = None

        return decoder_state, decoder_cell


    # Each forward pass of out network is defined for (batch_size , max_seq_size)
    def forward(self, x, y):

        x.to(device)
        y.to(device)

        num_layers_decoder = self.num_layers_decoder
        num_layers_encoder = self.num_layers_encoder
        batch_size = self.batch_size
        output_size = self.output_size
        hidden_dim = self.hidden_dim
        num_directions = 2 if self.bidirectional else 1
        SOS_TOKEN = 128

        # Calculate embedding first
        # (batch_size , max_sequence_length) -> (batch_size , max_sequence_length, embedding_dimension)
        x = self.embedding_encoder(x)

        if(self.encoder_cell_type == "LSTM") : encoder_output, (encoder_hidden, encoder_cell) = self.rnn_encoder(x)

        # hidden_state : (num_directions * num_layers , batch_size , hidden_state_size)
        else : 
            
            encoder_output, encoder_hidden = self.rnn_encoder(x)
            encoder_cell = None

        decoder_state, decoder_cell= self.initialize_decoder_state(self.encoder_cell_type, encoder_hidden, encoder_cell, self.decoder_cell_type, 
                                                                   self.bidirectional,  num_layers_decoder, num_layers_encoder)
        
        decoder_inputs = torch.full((batch_size, 1), SOS_TOKEN).to(device)

        decoder_outputs = torch.empty((self.max_seq_size, batch_size, self.output_size)).to(device)
        
        for t in range(self.max_seq_size):
            
            decoder_inputs = self.embedding_decoder(decoder_inputs.to(device))
        
            if self.decoder_cell_type == "LSTM":
                decoder_output, (decoder_state, decoder_cell) = self.rnn_decoder(decoder_inputs, (decoder_state.contiguous(), decoder_cell.contiguous()))
            else:
                decoder_output, decoder_state = self.rnn_decoder(decoder_inputs, decoder_state.contiguous())
                decoder_cell = None

            if(num_layers_decoder > 1 ) : decoder_output = self.dropout(decoder_output)

            decoder_outputs[t] = self.fc1(decoder_output).squeeze(dim=1)
            
            # Determine whether to use teacher forcing or predicted output
            use_teacher_forcing = random.random() < self.teacher_forcing_prob
            
            # Obtain the next input to the decoder
            if use_teacher_forcing:

                decoder_inputs = y[:, t].unsqueeze(0)  # Use ground truth input

            else:

                #print(decoder_outputs[t].shape)
                indices = torch.argmax(decoder_outputs[t], dim=1)

                #print(indices.shape)
                decoder_inputs = indices.unsqueeze(dim=1)
                #print("Non Teacher Forcing : ", decoder_inputs.shape)

            if (decoder_inputs.shape[0]!= batch_size) : decoder_inputs = decoder_inputs.transpose(0,1)

        decoder_outputs = decoder_outputs.transpose(0, 1)

        return decoder_outputs
    

def compare_sequences(batch1, batch2):
    """
    Compare two batches of sequences and return the number of sequences that are exactly the same.

    Args:
        batch1 (torch.Tensor): Batch of sequences of shape [batch_size, max_seq_length].
        batch2 (torch.Tensor): Batch of sequences of shape [batch_size, max_seq_length, vocab_size].

    Returns:
        int: Number of sequences that are exactly the same.
    """
    # Get the predicted sequences by finding the index of the maximum probability

    device = batch1.device
    batch2 = batch2.to(device)

    predicted_sequences = torch.argmax(batch2, dim=2)

    # Compare the predicted sequences with the ground truth sequences
    num_same_sequences = torch.sum(torch.all(batch1 == predicted_sequences, dim=1)).item()

    return num_same_sequences

def test_model_instance(configs):

    count = 0

    VOCAB_SIZE = 131
    BATCH_SIZE = 4
    MAX_SEQ_SIZE = 28

    source = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_SEQ_SIZE)).to(device)
    target = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_SEQ_SIZE)).to(device)

    configs = list(itertools.product(*configs.values()))

    for config in tqdm(configs):

        input_embedding_size, num_encoder_layers, num_decoder_layers, hidden_layer_size, cell_type_encoder, cell_type_decoder, bidirectional, dropout, teacher_forcing = config
    
        # Create an instance of the seq2seq model using the parameter values
        model = seq2seq(VOCAB_SIZE, input_embedding_size, hidden_layer_size, num_encoder_layers, num_decoder_layers,
                   dropout, bidirectional, cell_type_encoder, cell_type_decoder, teacher_forcing,
                   BATCH_SIZE, MAX_SEQ_SIZE, debugging=False).to(device)
        
        output = model(source, target)

        if(output.shape[0]==BATCH_SIZE and output.shape[1]==MAX_SEQ_SIZE and output.shape[2]==VOCAB_SIZE) : count+=1
        
    print("PASSED {} CONFIGS.".format(count))

import torch
import torch.nn as nn
import random

class seq2seq(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers_encoder, num_layers_decoder, 
                 dropout, bidirectional, encoder_cell_type, decoder_cell_type, teacher_forcing, 
                 batch_size, max_seq_size, debugging=False):

        super(seq2seq, self).__init__()

        self.output_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.encoder_cell_type = encoder_cell_type
        self.decoder_cell_type = decoder_cell_type
        self.teacher_forcing_prob = teacher_forcing
        self.debugging = debugging
        self.batch_size = batch_size
        self.max_seq_size = max_seq_size

        self.dropout = nn.Dropout(self.dropout_prob)
        self.embedding_encoder = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.embedding_dim).to(device)
        self.embedding_decoder = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.embedding_dim).to(device)

        self.rnn_encoder = self.cell(num_layers_encoder, encoder_cell_type, bool(self.bidirectional))
        self.rnn_decoder = self.cell(num_layers_decoder, decoder_cell_type, 0)

        # Attention layer
        self.attention = nn.Linear(hidden_dim, hidden_dim)

        # Final layer for calculating probabilities
        self.fc1 = nn.Linear(self.hidden_dim, self.output_size)

    def cell(self, num_layers, cell_type, bidirectional):

        # Defining Cells
        cells = {
            "LSTM": nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, 
                            dropout=self.dropout_prob, bidirectional=bool(bidirectional)).to(device),
            "GRU": nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, 
                          dropout=self.dropout_prob, bidirectional=bool(bidirectional)).to(device),
            "RNN": nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, 
                          dropout=self.dropout_prob, bidirectional=bool(bidirectional)).to(device)
        }

        return cells[cell_type]

    def initialize_decoder_state(self, encoder_cell_type, encoder_state, encoder_cell, decoder_cell_type, bidirectional,
                                 num_decoder_layers, num_encoder_layers):

        batch_size = encoder_state[0].size(0)  # Get the batch size from the encoder states

        if encoder_cell_type == "LSTM":
            if bidirectional:
                forward_state, backward_state = encoder_state[:num_encoder_layers], encoder_state[num_encoder_layers:]
                forward_cell, backward_cell = encoder_cell[:num_encoder_layers], encoder_cell[num_encoder_layers:]
                encoder_state = (torch.mean(forward_state, dim=0) + torch.mean(backward_state, dim=0)) / 2
                encoder_cell = (torch.mean(forward_cell, dim=0) + torch.mean(backward_cell, dim=0)) / 2
            else:
                encoder_state = torch.mean(encoder_state, dim=0)
                encoder_cell = torch.mean(encoder_cell, dim=0) if decoder_cell_type == "LSTM" else None
        else:
            forward_state = encoder_state[:num_encoder_layers]
            backward_state = encoder_state[num_encoder_layers:] if bidirectional else None
            encoder_state = (torch.mean(forward_state, dim=0) + torch.mean(backward_state, dim=0)) / 2 if bidirectional else torch.mean(forward_state, dim=0)
            encoder_cell = None

        decoder_state = encoder_state.unsqueeze(0).expand(num_decoder_layers, batch_size, -1)

        if decoder_cell_type == "LSTM":
            decoder_cell = decoder_state
        else:
            decoder_cell = None

        return decoder_state, decoder_cell

    def forward(self, x, y):
        x.to(device)
        y.to(device)

        num_layers_decoder = self.num_layers_decoder
        num_layers_encoder = self.num_layers_encoder
        batch_size = self.batch_size
        output_size = self.output_size
        hidden_dim = self.hidden_dim
        num_directions = 2 if self.bidirectional else 1
        SOS_TOKEN = 128

        x = self.embedding_encoder(x)

        if self.encoder_cell_type == "LSTM":
            encoder_output, (encoder_hidden, encoder_cell) = self.rnn_encoder(x)
        else:
            encoder_output, encoder_hidden = self.rnn_encoder(x)
            encoder_cell = None

        decoder_state, decoder_cell = self.initialize_decoder_state(self.encoder_cell_type, encoder_hidden, encoder_cell, 
                                                                     self.decoder_cell_type, self.bidirectional, 
                                                                     num_layers_decoder, num_layers_encoder)

        decoder_inputs = torch.full((batch_size, 1), SOS_TOKEN).to(device)
        decoder_outputs = torch.empty((self.max_seq_size, batch_size, self.output_size)).to(device)

        for t in range(self.max_seq_size):
            decoder_inputs = self.embedding_decoder(decoder_inputs.to(device))

            if self.decoder_cell_type == "LSTM":
                decoder_output, (decoder_state, decoder_cell) = self.rnn_decoder(decoder_inputs, (decoder_state.contiguous(), decoder_cell.contiguous()))
            else:
                decoder_output, decoder_state = self.rnn_decoder(decoder_inputs, decoder_state.contiguous())
                decoder_cell = None

            if num_layers_decoder > 1:
                decoder_output = self.dropout(decoder_output)

            # Attention mechanism
            attn_weights = torch.softmax(self.attention(encoder_output), dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)

            decoder_output = torch.cat((decoder_output, context), dim=1)

            decoder_outputs[t] = self.fc1(decoder_output).squeeze(dim=1)


            

            use_teacher_forcing = random.random() < self.teacher_forcing_prob

            if use_teacher_forcing:
                decoder_inputs = y[:, t].unsqueeze(0)
            else:
                indices = torch.argmax(decoder_outputs[t], dim=1)
                decoder_inputs = indices.unsqueeze(dim=1)

            if decoder_inputs.shape[0] != batch_size:
                decoder_inputs = decoder_inputs.transpose(0, 1)

        decoder_outputs = decoder_outputs.transpose(0, 1)

        return decoder_outputs

