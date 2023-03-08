"""Main script to test a pretrained model"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mvrss.utils.paths import Paths
from mvrss.utils.functions import count_params
from mvrss.learners.tester import Tester
from mvrss.models import TMVANet, MVNet, MVANet, MVA_DCN, TMVA_DCN, TMVA_TDC, PKCIn, PKCOn, AdaPKC
from mvrss.loaders.dataset import Carrada
from mvrss.loaders.dataloaders import SequenceCarradaDataset

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default='config.json')
    args = parser.parse_args()
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    paths = Paths().get()
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name
    model_path = path / 'results' / 'model.pt'
    test_results_path = path / 'results' / 'test_results.json'

    if cfg['model'] == 'mvnet':
        model = MVNet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
        print('Number of trainable parameters in the model: %s' % str(count_params(model)))
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    elif cfg['model'] == 'pkcin':
        # loading parallel saved model in local single gpu mode
        model = PKCIn(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
        print('Number of trainable parameters in the model: %s' % str(count_params(model)))
        saved_model = torch.load(model_path)
        model.load_state_dict({k.replace('module.',''):v for k, v in saved_model.items()})
        model = nn.DataParallel(model).cuda()
    elif cfg['model'] == 'pkcon':
        # loading parallel saved model in local single gpu mode
        model = PKCOn(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
        print('Number of trainable parameters in the model: %s' % str(count_params(model)))
        saved_model = torch.load(model_path)
        model.load_state_dict({k.replace('module.',''):v for k, v in saved_model.items()})
        model = nn.DataParallel(model).cuda()
    else:
        model = TMVANet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
        print('Number of trainable parameters in the model: %s' % str(count_params(model)))
        model.load_state_dict(torch.load(model_path))
        model.cuda()


    tester = Tester(cfg)
    print(tester)
    data = Carrada()
    test = data.get('Test')
    testset = SequenceCarradaDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    tester.set_annot_type(cfg['annot_type'])
    if cfg['model'] == 'mvnet':
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=False)
    else:
        # print(cfg['model'])
        test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=True)
    tester.write_params(test_results_path)

if __name__ == '__main__':
    test_model()
