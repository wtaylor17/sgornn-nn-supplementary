from sgornn import addition_problem
from sgornn import ExpRNNModel, FastRNNModel, VPRNNModel, LSTMModel, ExpRNNCell
from vprnn.layers import VanillaCell as VPRNNCell
from keras.layers import SimpleRNNCell
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument('--model-type', type=str, default='sgornn-v',
                    help='the model type. one of [sgornn-v, sgornn-e, exprnn, vprnn, fastrnn, lstm]')
parser.add_argument('--sequence-length', type=int, default=500,
                    help='sequence length (T)')
parser.add_argument('--epochs', type=int, default=75,
                    help='epochs (batches x 100) used to train')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='initial lr. decays to zero linearly.')
parser.add_argument('--model-output-path', type=str,
                    default=None,
                    help='if provided, save the model weights to this path')
parser.add_argument('--history-output-path',
                    type=str, default=None,
                    help='if provided, save the training history to this path')
parser.add_argument('--no-scalar-clip', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    model_type = ['sgornn-v', 'sgornn-e', 'exprnn', 'vprnn', 'fastrnn', 'lstm'].index(args.model_type)
    print(f'Model type: {args.model_type} ({model_type})')

    if args.model_type in ['sgornn-v', 'sgornn-e', 'fastrnn']:
        cell_type = ['sgornn-v', 'sgornn-e', 'fastrnn'].index(args.model_type)
        cell_type = [VPRNNCell, ExpRNNCell, SimpleRNNCell][cell_type]
        model = FastRNNModel(input_dim=2,
                             output_dim=1,
                             layers=1,
                             clip_scalar=not args.no_scalar_clip,
                             dim=128,
                             rots=7,
                             output_activation='linear',
                             cell_class=cell_type)
    else:
        model_type = ['exprnn', 'vprnn', 'lstm'].index(args.model_type)
        model = [ExpRNNModel, VPRNNModel, LSTMModel][model_type](input_dim=2,
                                                                 output_dim=1,
                                                                 layers=1,
                                                                 clip_scalar=not args.no_scalar_clip,
                                                                 dim=128,
                                                                 rots=7,
                                                                 output_activation='linear')
    model.summary()

    history = addition_problem.fit(model,
                                   args.sequence_length,
                                   epochs=args.epochs,
                                   initial_lr=args.lr).history

    if args.model_output_path:
        model.save_weights(args.model_output_path)

    if args.history_output_path:
        json.dump(history, open(args.history_output_path, 'w'))
