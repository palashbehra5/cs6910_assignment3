CS6910 - Assignment 3

This repository contains the code for CS6910 - Assignment 3. The assignment involves implementing a sequence-to-sequence model with attention for a specific task.
Usage

To run the code, you need to have Python installed on your system. The code is written in Python 3.
```
git clone <repository_url>
```
Install the required dependencies:
```
pip install -r requirements.txt
```
Run the code:

    python train.py [arguments]

Arguments

The following arguments can be passed to the script:

    -wp, --wandb_project: Specify the WandB project name (default: "myprojectname").
    -we, --wandb_entity: Specify the WandB entity name (default: "myname").
    -ec, --encoder_cell: Specify the type of encoder cell (default: "GRU").
    -dc, --decoder_cell: Specify the type of decoder cell (default: "LSTM").
    -e, --epochs: Specify the number of training epochs (default: 25).
    -es, --embedding_size: Specify the size of the embedding layer (default: 256).
    -nle, --num_layers_encoder: Specify the number of layers in the encoder (default: 3).
    -nld, --num_layers_decoder: Specify the number of layers in the decoder (default: 3).
    -sz, --hidden_size: Specify the hidden size of the encoder and decoder (default: 256).
    -b, --batch_size: Specify the batch size for training (default: 64).
    -bi, --bi_directional: Specify whether to use a bi-directional encoder (default: 1).
    -a, --attention: Specify whether to use attention mechanism (default: 0).
    -p, --dropout: Specify the dropout probability (default: 0.2).
    -tf, --teacher_forcing: Specify the teacher forcing ratio (default: 0.75).

Note: All arguments are optional. If not specified, the default values mentioned above will be used.
Configuration

The script uses the following configurations:

    VOCAB_SIZE: The vocabulary size (default: 131).
    MAX_SEQ_SIZE: The maximum sequence size (default: 28).

Model Parameters

The model parameters are defined as follows:
```
model_params = {
    "encoder_cell": args.encoder_cell,
    "decoder_cell": args.decoder_cell,
    "embedding_size": args.embedding_size,
    "num_layers_encoder": args.num_layers_encoder,
    "num_layers_decoder": args.num_layers_decoder,
    "hidden_size": args.hidden_size,
    "bi_directional": args.bi_directional,
    "dropout": args.dropout,
    "vocab_size": VOCAB_SIZE,
    "max_seq_size": MAX_SEQ_SIZE,
    "attention": args.attention
}
```
Training Parameters

The training parameters are defined as follows:
```
training_params = {
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "teacher_forcing": args.teacher_forcing
}
```
WandB Parameters

The WandB parameters are defined as follows:
```
wandb_params = {
    "project_name": args.wandb_project,
    "entity_name": args.wandb_entity
}
```
Make sure to adjust the parameters according to your requirements before running the code.
