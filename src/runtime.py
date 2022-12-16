import pandas as pd
import numpy as np
from .model import GAIN
from .missingness import MissingMechanism
from .evaluator import Evaluator
import os


def run(args: dict, data):
    """Run the entire GAIN pipeline-training

    Args:
        args (dict): arguments dictionary (for all arguments see README.md)
        data (numpy array): numpy array (!) with non-normalized data

    Return:
        imputed_data, model : returns imputed data and trained GAIN object
    """
    model = GAIN(args)
    data_copy = data.copy()
    ori_mask = 1 - np.isnan(
        data_copy
    )  # 1 indicates present data, 0 missing in the training data (MAKE SURE missing values are np.nan)
    data_copy[ori_mask == 0] = 0.0
    missingness_prop = np.sum(1 - ori_mask) / np.prod(data.shape)
    # introduce missingness for training
    missing_mech = MissingMechanism(args, data)
    # UNCOMMENT FOR OLD MISSING MECHANISM
    # (miss_data, mask,) = model.create_missing(data, args["missing_rate"])
    # UNCOMMENT FOR NEW MISSING MECHANISM
    miss_data, mask = missing_mech.missingness_pipeline()
    # model.create_missing(data, args["missing_rate"]) #missing_mech.missingness_pipeline()
    final_mask = ori_mask * mask  # new easier method to calculate combined mask
    miss_data[final_mask == 0] = 0.0

    imputed_data = model.train(
        miss_data, final_mask, data_copy
    )  # use missing data and final mask to train GAIN, original data used for evaluation
    if missingness_prop > 0:  # warn about invalid RMSE metrics
        print(
            f"### WARNING! RMSE metrics invalid, due to {missingness_prop:.4f} missing data in train set"
        )
    if len(imputed_data) != 2:  # no collapse
        ohe_idx = model.create_rmse_idxs(
            data_copy, args["OHE_features"], missing_mech.num_cols
        )
        rmse = model.RMSE(data_copy, imputed_data, final_mask, ohe_idx)
        print(f"Final RMSE performance {rmse}")

    else:  # if collapse happens tuple of return different
        print("RMSE cannot be calculated")
        model = imputed_data[1]
        imputed_data = imputed_data[0]
    return (
        imputed_data,
        model,
    )  # return imputed data and the model (if no collapse its the GAIN object)


if __name__ == "__main__":
    data_path = "data/letter.csv"
    df = pd.read_csv(data_path).values
    dct_args = {
        "batch_size": 128,
        "hint_rate": 0.9,
        "alpha": 10,
        "beta": 10,
        "epochs": 70,
        "missing_rate": 0.2,
        "OHE_features": False,  # letter_ohe: (15,31) #letter_3ohe: (13,29), (29, 45), (45, 61)
        "learning_rate": 1e-4,
        "log": True,
        "debug": False,
        "rounding": True,
        "early_stop": False,
        "normalized_loss": True,
        "validation": False,
    }

    df_new, mask_ = GAIN.create_missing(df, 0.1, np.nan)

    imputed, rmse = run(dct_args, df)

    path_to_log = (
        os.getcwd()
        + "/logs/train-run-"
        + input("What was the time on the log file? /logs/train-run-")
    )
    ev = Evaluator(path_to_log)
    ev.print_summary()
    ev.make_plot(["Generator", "Discriminator"], ["RMSE"]).write_image(
        f"{os.getcwd()}/logs/training_plot.png"
    )

