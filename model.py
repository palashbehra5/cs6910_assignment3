import torch
import torch.nn as nn
import random
import itertools
import torch.optim as optim
from tqdm import tqdm
import pickle

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

def test_model_instance(configs, mode, batch_size):

    count = 0

    VOCAB_SIZE = 131
    BATCH_SIZE = batch_size
    MAX_SEQ_SIZE = 28

    source = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_SEQ_SIZE)).to(device)
    target = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, MAX_SEQ_SIZE)).to(device)

    configs = list(itertools.product(*configs.values()))

    for config in tqdm(configs):

        input_embedding_size, num_encoder_layers, num_decoder_layers, hidden_layer_size, cell_type_encoder, cell_type_decoder, bidirectional, dropout, teacher_forcing = config
    
        # Create an instance of the seq2seq model using the parameter values
        if(mode=='attn'):

            model = seq2seq_attn(VOCAB_SIZE, input_embedding_size, hidden_layer_size, num_encoder_layers, num_decoder_layers,
                   dropout, bidirectional, cell_type_encoder, cell_type_decoder, teacher_forcing,
                   BATCH_SIZE, MAX_SEQ_SIZE, debugging=False).to(device)

        else:

            model = seq2seq(VOCAB_SIZE, input_embedding_size, hidden_layer_size, num_encoder_layers, num_decoder_layers,
                   dropout, bidirectional, cell_type_encoder, cell_type_decoder, teacher_forcing,
                   BATCH_SIZE, MAX_SEQ_SIZE, debugging=False).to(device)
        
        output = model(source, target)

        if(output.shape[0]==BATCH_SIZE and output.shape[1]==MAX_SEQ_SIZE and output.shape[2]==VOCAB_SIZE) : count+=1
        
    print("PASSED {} CONFIGS.".format(count))

class seq2seq_attn(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers_encoder, num_layers_decoder, 
                 dropout, bidirectional, encoder_cell_type, decoder_cell_type, teacher_forcing, 
                 batch_size, max_seq_size, debugging=False):

        super(seq2seq_attn, self).__init__()

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
        num_directions_encoder = 1 if bidirectional == 0 else 2
        self.attention = nn.Linear(hidden_dim, hidden_dim)  # Attention layer
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.output_size)  # Output layer

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

        # Calculate embedding first
        x = self.embedding_encoder(x)

        if self.encoder_cell_type == "LSTM":
            encoder_output, (encoder_hidden, encoder_cell) = self.rnn_encoder(x)
        else:
            encoder_output, encoder_hidden = self.rnn_encoder(x)
            encoder_cell = None

        # Modify the encoder_hidden based on bidirectional flag
        if self.bidirectional:

            forward_output = encoder_output[:, :, :self.hidden_dim]
            backward_output = encoder_output[:, :, self.hidden_dim:]
            encoder_output = forward_output + backward_output

        decoder_state, decoder_cell = self.initialize_decoder_state(
            self.encoder_cell_type, encoder_hidden, encoder_cell, self.decoder_cell_type,
            self.bidirectional, num_layers_decoder, num_layers_encoder
        )

        decoder_inputs = torch.full((batch_size, 1), SOS_TOKEN).to(device)
        decoder_outputs = torch.empty((self.max_seq_size, batch_size, self.output_size)).to(device)

        for t in range(self.max_seq_size):

            

            decoder_inputs = self.embedding_decoder(decoder_inputs.to(device))

            if self.decoder_cell_type == "LSTM":
                decoder_output, (decoder_state, decoder_cell) = self.rnn_decoder(
                    decoder_inputs, (decoder_state.contiguous(), decoder_cell.contiguous())
                )
            else:
                decoder_output, decoder_state = self.rnn_decoder(decoder_inputs, decoder_state.contiguous())
                decoder_cell = None

            if num_layers_decoder > 1:
                decoder_output = self.dropout(decoder_output)

            attn_weights = torch.matmul(self.attention(encoder_output), decoder_output.transpose(1, 2))

            attn_weights = attn_weights.squeeze(dim=2)

            attn_weights = torch.softmax(attn_weights, dim=1)

            context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(dim=1)

            decoder_output = decoder_output.squeeze(dim=1)

            decoder_output = torch.cat((decoder_output, context_vector), dim=1)
            decoder_outputs[t] = self.fc1(decoder_output).squeeze(dim=1)

            # Determine whether to use teacher forcing or predicted output
            use_teacher_forcing = random.random() < self.teacher_forcing_prob

            # Obtain the next input to the decoder
            if use_teacher_forcing:
                decoder_inputs = y[:, t].unsqueeze(0)  # Use ground truth input
            else:
                indices = torch.argmax(decoder_output, dim=1)
                decoder_inputs = indices.unsqueeze(dim=1)

            if decoder_inputs.shape[0] != batch_size:
                decoder_inputs = decoder_inputs.transpose(0, 1)

        decoder_outputs = decoder_outputs.transpose(0, 1)
        return decoder_outputs
    
# This does not seem to work
# Go for standard implementation
##################################################################################   
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html#
##################################################################################

class Attention(nn.Module):

    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
    

def train(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER, 
                 DROPOUT, BIDIRECTIONAL, CELL_TYPE_ENCODER, CELL_TYPE_DECODER, TEACHER_FORCING, 
                 BATCH_SIZE, MAX_SEQ_SIZE, EPOCHS, train_dataset, val_dataset) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = seq2seq(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER, 
                    DROPOUT, BIDIRECTIONAL, CELL_TYPE_ENCODER, CELL_TYPE_DECODER, TEACHER_FORCING, 
                    BATCH_SIZE, MAX_SEQ_SIZE, debugging = False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        
        model.train()
        running_loss = 0.0
        train_accuracy = 0
        val_accuracy = 0
        
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, targets)
            
            train_accuracy += compare_sequences(targets, outputs)

            loss = criterion(outputs.reshape(-1, model.output_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
            
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs, targets)

                loss = criterion(outputs.reshape(-1, model.output_size), targets.reshape(-1))
                val_accuracy += compare_sequences(targets, outputs)
                
                val_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {val_loss / len(val_loader)}")
        print("Training Accuracy {0}, Validation Accuracy {1}".format(train_accuracy/(len(train_dataset)), val_accuracy/(len(val_dataset))))
        torch.cuda.empty_cache()

    return model

def test(model, test_dataset):

    predictions, targets = [], []

    with open("idx_to_char.pickle", "rb") as file:
        idx_to_char = pickle.load(file)

    device = model.device
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False)
    acc = 0

    for src,target in test_loader:

        src = src.to(device)
        target = target.to(device)

        pred = model(src)
        acc += compare_sequences(target, pred)

        predicted_sequences = torch.argmax(pred, dim=2)

        for seq in predicted_sequences:

            predicted_word = ''.join([idx_to_char[idx] for idx in seq if (idx != 128 or idx != 129 or idx != 130 )])
            predictions.append(predicted_word)

        for seq in target:

            predicted_word = ''.join([idx_to_char[idx] for idx in seq if (idx != 128 or idx != 129 or idx != 130 )])
            targets.append(predicted_word)

    print("Testing accuracy for model : {}".format(acc/len(test_dataset)))

    return predictions, targets

