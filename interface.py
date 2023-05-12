import argparse

VOCAB_SIZE = 131
MAX_SEQ_SIZE = 28

def parse():

  parser = argparse.ArgumentParser(description='CS6910 - Assignment 3 :')

  # String type arguments
  parser.add_argument('-wp','--wandb_project', type=str, default = "myprojectname")
  parser.add_argument('-we','--wandb_entity', type=str, default = "myname")
  parser.add_argument('-ec','--encoder_cell', type=str, default = "GRU")
  parser.add_argument('-dc','--decoder_cell', type=str, default = "LSTM")

  # Integer type arguments
  parser.add_argument('-e','--epochs', type=int, default = 25)
  parser.add_argument('-es','--embedding_size', type=int, default = 256)
  parser.add_argument('-nle','--num_layers_encoder', type=int, default = 3)
  parser.add_argument('-nld','--num_layers_decoder', type=int, default = 3)
  parser.add_argument('-sz','--hidden_size', type=int, default = 256)
  parser.add_argument('-b','--batch_size', type=int, default = 64)
  parser.add_argument('-bi','--bi_directional', type=int, default = 1)
  parser.add_argument('-a','--attention', type=int, default = 0)

  # Float type arguments
  parser.add_argument('-p','--dropout',type=float, default = 0.2)
  parser.add_argument('-tf','--teacher_forcing',type=float, default = 0.75)

  args = parser.parse_args()

  return args

args = parse()

model_params = {

    "encoder_cell" : args.encoder_cell,
    "decoder_cell" : args.decoder_cell,
    "embedding_size" : args.embedding_size,
    "num_layers_encoder" : args.num_layers_encoder,
    "num_layers_decoder" : args.num_layers_decoder,
    "hidden_size" : args.hidden_size,
    "bi_directional" : args.bi_directional,
    "dropout" : args.dropout,
    "vocab_size" : VOCAB_SIZE,
    "max_seq_size" : MAX_SEQ_SIZE,
    "attention" : args.attention

}


training_params = {

    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "teacher_forcing" : args.teacher_forcing

}

wandb_params = {

    "project_name": args.wandb_project,
    "entity_name": args.wandb_entity

}
