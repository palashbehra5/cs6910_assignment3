import torch.nn as nn
import torch
import random

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type, bidirectional):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.cell_type = cell_type.upper()
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(self.dropout_prob)
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embedding_dim)

        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob, bidirectional=bool(self.bidirectional))
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob, bidirectional=bool(self.bidirectional))
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob, bidirectional=bool(self.bidirectional))
        else:
            raise ValueError(f"Unsupported cell_type '{self.cell_type}'. Supported types: 'RNN', 'LSTM', 'GRU'.")

    def forward(self, x):
        # x has shape (batch_size, seq_len)

        # Calculate embedding
        embedding = self.dropout(self.embedding(x))

        # Pass embedding through RNN
        output, hidden = self.rnn(embedding)

        # Apply dropout to hidden state
        if self.cell_type == 'LSTM':
            hidden = tuple([self.dropout(h) for h in hidden])
        else:
            hidden = self.dropout(hidden)

        return hidden


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type, bidirectional, teacher_forcing):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.cell_type = cell_type.upper()
        self.bidirectional = bidirectional
        self.teacher_forcing_prob = teacher_forcing

        self.dropout = nn.Dropout(self.dropout_prob)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob, bidirectional=bool(self.bidirectional))
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob, bidirectional=bool(self.bidirectional))
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob, bidirectional=bool(self.bidirectional))
        else:
            raise ValueError(f"Unsupported cell_type '{self.cell_type}'. Supported types: 'RNN', 'LSTM', 'GRU'.")

        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, x, hidden):
        # x has shape (batch_size, seq_len)
        # teacher_forcing_prob is the probability of using teacher forcing
        teacher_forcing_prob = self.teacher_forcing_prob

        # Calculate embedding
        embedding = self.embedding(x)

        print(x.shape)
        # Initialize outputs tensor
        batch_size, seq_len, _ = x.shape

        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(x.device)

        # Initialize input word
        input_word = embedding[:, 0, :]

        # Loop over sequence length
        for t in range(1, seq_len):
            # Decide whether to use teacher forcing or not
            use_teacher_forcing = random.random() < teacher_forcing_prob

            # Use previous output as input if teacher forcing is not used
            if not use_teacher_forcing:
                input_word = output.argmax(dim=2)[:, t-1]
                input_word = self.embedding(input_word)

            # Pass input word and hidden state through RNN
            output, hidden = self.rnn(input_word.unsqueeze(1), hidden)

            # Apply dropout to output
            output = self.dropout(output)

            # Pass output through linear layer
            output = self.fc(output.squeeze(1))

            # Append output to outputs tensor
            outputs[:, t, :] = output

            # Update input word
            if use_teacher_forcing:
                input_word = embedding[:, t, :]

        return outputs, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
    def forward(self, src, trg):
        # src shape: (src_len, batch_size)
        # trg shape: (trg_len, batch_size)
        # teacher_forcing_ratio is the probability to use teacher forcing
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
        teacher_forcing_ratio = self.teacher_forcing_ratio
        
        # initialize output tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # encoder output
        encoder_output, hidden = self.encoder(src)
        
        # decoder input is always <sos>
        input = trg[0, :]
        
        for t in range(1, trg_len):
            # decoder output
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            
            # decide whether to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our outputs
            top1 = output.argmax(1)
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force and t < trg_len else top1
        
        return outputs


