import os
import yaml


def get_config(dir='config/config_WPDP.yaml'):
    "This function is used to load the parameter settings in the config.yaml file"
    # add direction join function when parse the yaml file
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    # add string concatenation function when parse the yaml file
    def concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    yaml.add_constructor('!join', join)
    yaml.add_constructor('!concat', concat)
    with open(dir, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

