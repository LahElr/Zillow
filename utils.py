import json
import os
import time
import logging
from typing import Any, Dict

# from catboost import CatBoostRegressor, Pool
import lightgbm
from catboost import CatBoostRegressor


def get_cur_time() -> str:
    r'''
    This function returns a string of current time in format `%Y%m%d_%H%M%S`
    '''
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


config = json.load(open(r"./config.json", "r"))
config['save_path'] = os.path.join(config['save_path'], config["exp_name"],
                                   get_cur_time())
if not os.path.exists(config["save_path"]):
    os.makedirs(config['save_path'])

# This piece of complicated code is necessary just so that matplotlib would not mess up logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(config["exp_name"])
logger_ch = logging.FileHandler(filename=f"{config['save_path']}/log.log",
                                mode="w")
logger_ch.setLevel(logging.NOTSET)
logger_ch.setFormatter(
    logging.Formatter(
        fmt=
        "%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(logger_ch)
logger.setLevel(logging.NOTSET)
logger.propagate = False
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL").propagate = False
logging.getLogger("urllib3").propagate = False


def read_config(url: str = None):
    r'''
    This function returns the requested config item, or the whole config file if `url` is None
    example of `url`: "train.optim.lr"
    '''
    if url is None:
        return config
    else:
        url = url.split(".")
        target = config
        for item in url:
            try:
                target = target[item]
            except KeyError:
                raise KeyError(
                    f"utils.read_config: your requested config item {url} does not exist!"
                )
        return target


def save_config():
    r"""
    This function would save the current config into the `config.save_path`
    """
    path_to_save = os.path.join(config["save_path"], "config.json")
    if not os.path.exists(config["save_path"]):
        os.makedirs(config['save_path'])
        save_config()
    json.dump(config, open(path_to_save, "w"))

if config["save_config"]:
    save_config()

r"""
try:
    import torch

    def save_model(model: torch.nn.Module, name: str = None) -> str:
        r'''
        Save a model, if `name` is not specified, the time of saving will be the name
        Return the name of the saved checkpoint.
        '''
        if not os.path.exists(config["save_path"]):
            os.makedirs(config['save_path'])
        if name is None:
            name = get_cur_time()
        torch.save(model.state_dict(),
                   os.path.join(config['save_path'], f"{name}.ckpt"))
        # model.load_state_dict(torch.load(PATH))
        return name
    
    def load_state_dict(name:str=None) -> Dict[str,Any]:
        r'''
        Load the model saved by `save_model` IN THIS RUNNING SESSION
        Return the state dict
        '''
        if not os.path.exists(os.path.join(config["save_path"],f"{name}.ckpt")):
            raise FileNotFoundError(f"The request checkpoint '{name}' cannot be found in this session. Please check.")
        return torch.load(os.path.join(config["save_path"],f"{name}.ckpt"))

except ModuleNotFoundError:
    logger.warning(f"PyTorch related functions not defines for no torch module found.")

"""

def save_lightgbm_model(model,name=None):
    if name is None:
        name = "model.model"
    
    model.save_model(os.path.join(config['save_path'],name))

def load_lightgbm_model(path):
    return lightgbm.Booster(model_file=path)

def save_catboost_model(model,name=None):
    if name is None:
        name = "model.model"
    
    model.save_model(os.path.join(config['save_path'],name))

def load_catboost_model(path):
    model = CatBoostRegressor()
    model.load_model(path)
    return model