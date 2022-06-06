from sgornn import har2
from sgornn.models import FastRNNModel, ExpRNNModel, VPRNNModel, LSTMModel
from sgornn.layers import ExpRNNCell
from vprnn.layers import VanillaCell as VPRNNCell
from keras.layers import SimpleRNNCell
from argparse import ArgumentParser
import json
from math import log2, ceil
import numpy as np

parser = ArgumentParser()
parser.add_argument('--model-type', type=str, default='sgornn-v',
                    help='the model type. one of [sgornn-v, sgornn-e, exprnn, vprnn, fastrnn, lstm]')
parser.add_argument('--units', type=int, default=128,
                    help='hidden state size (h)')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers (L)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate.')
parser.add_argument('--model-output-path', type=str,
                    default=None,
                    help='if provided, save the model weights to this path')
parser.add_argument('--history-output-path',
                    type=str, default=None,
                    help='if provided, save the training history to this path')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--no-scalar-clip', action='store_true')
parser.add_argument('--seed', type=int, default=42)

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)

    model_type = ['sgornn-v', 'sgornn-e', 'exprnn', 'vprnn', 'fastrnn', 'lstm'].index(args.model_type)
    print(f'Model type: {args.model_type} ({model_type})')

    if args.model_type in ['sgornn-v', 'sgornn-e', 'fastrnn']:
        cell_type = ['sgornn-v', 'sgornn-e', 'fastrnn'].index(args.model_type)
        cell_type = [VPRNNCell, ExpRNNCell, SimpleRNNCell][cell_type]
        model = FastRNNModel(input_dim=9,
                             output_dim=1,
                             layers=args.layers,
                             clip_scalar=not args.no_scalar_clip,
                             dim=args.units,
                             rots=ceil(log2(args.units)),
                             output_activation='sigmoid',
                             cell_class=cell_type)
    else:
        model_type = ['exprnn', 'vprnn', 'lstm'].index(args.model_type)
        model = [ExpRNNModel, VPRNNModel, LSTMModel][model_type](input_dim=9,
                                                                 output_dim=1,
                                                                 layers=args.layers,
                                                                 clip_scalar=not args.no_scalar_clip,
                                                                 dim=args.units,
                                                                 rots=7,
                                                                 output_activation='sigmoid')

    model.summary()


    def schedule(epoch, _):
        if epoch >= 200:
            return args.lr / 10
        else:
            return args.lr


    history = har2.fit(model, lr=args.lr,
                       lr_scheduler=schedule,
                       epochs=300,
                       batch_size=args.batch_size).history

    if args.model_output_path:
        model.save_weights(args.model_output_path)

    if args.history_output_path:
        history = {k: list(map(float, v))
                   for k, v in history.items()}
        with open(args.history_output_path, 'w') as fp:
            fp.write(json.dumps(history))
