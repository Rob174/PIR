# Extracted from enet library
from tensorflow.keras.layers import Input,Reshape,Activation
from tensorflow.keras import Model
import os
from enet.models.enet_unpooling import encoder, decoder
import pickle as pkl


def transfer_weights(model, weights, keep_top=False):
    """
    Transfers weights from torch-enet if they are available as {PROJECT_ROOT}/models/pretrained/torch_enet.pkl after
    running from_torch.py.

    :param keep_top: Skips the final Transpose Convolution layer if False.
    :param model: the model to copy the weights to.
    :param weights: the filename that contains the set of layers to copy. Run from_torch.py first.
    :return: a model that contains the updated weights. This function mutates the contents of the input model as well.
    """

    def special_cases(idx):
        """
        Handles special cases due to non-matching layer sequences
        :param idx: original index of layer
        :return: the corrected index of the layer as well as the corresponding layer
        """
        idx_mapper = {
            266: 267,
            267: 268,
            268: 266,
            299: 300,
            300: 301,
            301: 299
        }

        actual_idx = idx_mapper[idx] if idx in idx_mapper else idx
        return actual_idx, model.layers[actual_idx]

    if not os.path.isfile(weights):
        print('ENet has found no compatible pretrained weights! Skipping weight transfer...')
    else:
        weights = os.path.abspath(weights)
        print('Loading pretrained weights from {}'.format(weights))
        with open(weights, 'rb') as fin:
            weights_mem = pkl.load(fin)
            idx = 0
            for num, layer in enumerate(model.layers):
                # handle special cases due to non-matching layer sequences
                actual_num, layer = special_cases(num)

                if not layer.weights:
                    continue

                item = weights_mem[idx]
                layer_name = item['torch_typename']
                new_values = layer.get_weights()
                if layer_name in ['cudnn.SpatialConvolution',
                                  'nn.SpatialDilatedConvolution']:
                    if 'bias' in item:
                        new_values = [item['weight'], item['bias']]
                    else:
                        new_values = [item['weight']]
                elif layer_name == 'nn.SpatialBatchNormalization':
                    new_values = [item['gamma'], item['beta'],
                                  item['moving_mean'], item['moving_variance']]
                elif layer_name == 'nn.PReLU':
                    new_values = [item['weight']]
                elif layer_name == 'nn.SpatialFullConvolution':
                    if keep_top:
                        if 'bias' in item:
                            new_values = [item['weight'], item['bias']]
                        else:
                            new_values = [item['weight']]
                else:
                    print('Unhandled layer type "{}"'.format(layer_name))
                layer.set_weights(new_values)
                idx += 1
    return model

def build(nb_couches_de_sortie_decodeur, w, h):
    inp = Input(shape=(h, w, 3), name='image')
    enet = encoder.build(inp)
    enet = decoder.build(enet, nc=nb_couches_de_sortie_decodeur)

    enet = Activation('softmax', name='output')(enet)
    model = Model(inputs=inp, outputs=enet)

    return model

def create(width,height,nb_classes):
    model = build(w=width,h=height,nb_couches_de_sortie_decodeur=nb_classes)
    model = transfer_weights(model,weights="/scratch/rmoine/PIR/torch_enet.pkl")
    print(model.summary())
    return model

if __name__ == "__main__":
    print(create(200,200,2).summary())