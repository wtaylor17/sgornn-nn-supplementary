from sgornn import ptb
from sgornn import FastRNNModel, VPRNNModel, ExpRNNModel, LSTMModel, ExpRNNCell
from vprnn.layers import VanillaCell as VPRNNCell
from keras.layers import Embedding, Dropout, SimpleRNNCell
from keras import Sequential
from argparse import ArgumentParser
from math import log2, ceil
import numpy as np

parser = ArgumentParser()
parser.add_argument('--model-type', type=str, default='sgornn',
                    help='the model type. one of [sgornn-v, sgornn-e, exprnn, vprnn, fastrnn, lstm]')
parser.add_argument('--units', type=int, default=256,
                    help='hidden state size (h)')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers (L)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate.')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--no-scalar-clip', action='store_true')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sequence-length', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.0)

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    embedding_layer = Embedding(10_000, args.units)

    model_type = ['sgornn-v', 'sgornn-e', 'exprnn', 'vprnn', 'fastrnn', 'lstm'].index(args.model_type)
    print(f'Model type: {args.model_type} ({model_type})')

    if args.model_type in ['sgornn-v', 'sgornn-e', 'fastrnn']:
        cell_type = ['sgornn-v', 'sgornn-e', 'fastrnn'].index(args.model_type)
        cell_type = [VPRNNCell, ExpRNNCell, SimpleRNNCell][cell_type]
        rnn_model = FastRNNModel(input_dim=args.units,
                                 output_dim=10_000,
                                 output_activation='softmax',
                                 clip_scalar=not args.no_scalar_clip,
                                 layers=args.layers,
                                 dim=args.units,
                                 return_sequences=True,
                                 rots=ceil(log2(args.units)),
                                 cell_class=cell_type)
    else:
        model_type = ['exprnn', 'vprnn', 'lstm'].index(args.model_type)
        rnn_model = [ExpRNNModel, VPRNNModel, LSTMModel][model_type](input_dim=args.units,
                                                                     output_dim=10_000,
                                                                     output_activation='softmax',
                                                                     clip_scalar=not args.no_scalar_clip,
                                                                     layers=args.layers,
                                                                     dim=args.units,
                                                                     return_sequences=True,
                                                                     rots=ceil(log2(args.units)))
    print(rnn_model)
    model = Sequential()
    model.add(embedding_layer)
    if args.dropout:
        model.add(Dropout(args.dropout))
    for layer in rnn_model.layers:
        model.add(layer)
    model.summary()


    def schedule(epoch, _):
        if epoch >= 200:
            return args.lr / 10
        else:
            return args.lr


    ptb.fit(model,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_steps=args.sequence_length)

    print('TEST EVAL: ', ptb.evaluate(model,
                                      batch_size=args.batch_size,
                                      num_steps=args.sequence_length))

