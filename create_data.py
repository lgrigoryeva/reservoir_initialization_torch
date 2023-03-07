"""Script to create training, validation and test data."""
import numpy as np

from utils import BrusselatorDataset, config

# For reproduceability
np.random.seed(42)


def main(config):

    # config["DATA"]["n_train"] = 100
    # config["DATA"]["n_val"] = 5
    # config["DATA"]["n_test"] = 5
    # config["DATA"]["l_trajectories"] = 500

    config["DATA"]["n_train"] = 400
    config["DATA"]["n_val"] = 50
    config["DATA"]["n_test"] = 50
    config["DATA"]["l_trajectories"] = 100
    config["DATA"]["l_trajectories_test"] = 500

    dataset_train = BrusselatorDataset(config["DATA"]["n_train"], config["DATA"]["l_trajectories"])
    dataset_val = BrusselatorDataset(config["DATA"]["n_val"], config["DATA"]["l_trajectories"])
    dataset_test = BrusselatorDataset(config["DATA"]["n_test"], config["DATA"]["l_trajectories_test"])

    dataset_train.save_data(path=config["PATH"], filename="training_data.npz")
    dataset_val.save_data(path=config["PATH"], filename="validation_data.npz")
    dataset_test.save_data(path=config["PATH"], filename="test_data.npz")


if __name__ == "__main__":
    main(config)
