
import hydra
from omegaconf import DictConfig, OmegaConf
from create_loaders import prepare_data


@hydra.main(config_path = "conf", config_name = "base", version_base = None)
def main(cfg : DictConfig) :
    
    # Parsing config file and printing it
    print(OmegaConf.to_yaml(cfg))

    # Preparing dataset for all clients
    train_loaders, validation_loaders, test_loaders = prepare_data(num_partitions = cfg.num_clients,
                                                                   batch_size=cfg.train_batch_size,
                                                                   val_ratio=cfg.val_ratio)
    
    # 


if __name__ == "__main__" :
    main()