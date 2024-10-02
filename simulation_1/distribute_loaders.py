import hydra
from omegaconf import DictConfig, OmegaConf
from create_loaders import prepare_data


@hydra.main(config_path = "conf", config_name = "base", version_base = None)
def distribute(cfg : DictConfig, client_num : int) :
    train_loaders, val_loaders, test_loader = prepare_data(num_partitions = cfg.num_clients,
                                                            batch_size=cfg.train_batch_size,
                                                            val_ratio=cfg.val_ratio)
    

    # Returning train and validation sets for that client
    return train_loaders[client_num], val_loaders[client_num]