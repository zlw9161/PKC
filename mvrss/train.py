"""Main script to train a model"""
import argparse
import json
from mvrss.utils.functions import count_params
from mvrss.learners.initializer import Initializer
from mvrss.learners.model import Model
from mvrss.models import TMVANet, MVNet, MVANet, MVA_DCN, TMVA_DCN, TMVA_TDC, PKCIn, PKCOn, AdaPKC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    init = Initializer(cfg)
    data = init.get_data()
    if cfg['model'] == 'mvnet':
        net = MVNet(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'mvanet':
        net = MVANet(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'mva_dcn':
        net = MVA_DCN(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tmva_dcn':
        net = TMVA_DCN(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tmva_tdc':
        net = TMVA_TDC(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin':
        net = PKCIn(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'adapkc':
        net = AdaPKC(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcon':
        net = PKCOn(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    else:
        net = TMVANet(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'])

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    if cfg['model'] == 'mvnet':
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)

if __name__ == '__main__':
    main()
