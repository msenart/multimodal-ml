import os
from MMSA import MMSA_run, get_config_regression

config = get_config_regression('lf_dnn', 'mosi')
config['featurePath'] = os.path.expanduser('~/mmsa_project/data/unaligned_50.pkl')
MMSA_run('lf_dnn', 'mosi', config=config, seeds=[1111], gpu_ids=[0])
