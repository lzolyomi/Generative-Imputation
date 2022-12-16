import numpy as np
from tqdm import tqdm


from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as kb

from .utils import (
    uniform_sampler,
    binary_sampler,
    create_C_total,
    create_C_i_per_feature,
)
from .callbacks import Callbacks
from .missingness import MissingMechanism


class GAIN:
    def __init__(self, args) -> None:
        """Initialize GAIN object with all arguments

        Args:
            args (dict): arguments dictionary

        Raises:
            TypeError: If validation set not specified correctly 
            (see README on order and contents)
        """
        self.args = args  # store all arguments
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.hint = args["hint_rate"]
        self.batch_size = args["batch_size"]
        self.epochs = args["epochs"]
        self.h_dim = 10  # NOTE: currently not used,
        self.lr = args["learning_rate"]
        self.rounding = args["rounding"] if "rounding" in args.keys() else False
        self.debug = args["debug"] if "debug" in args.keys() else False
        self.validation = args["validation"] if "validation" in args.keys() else False
        self.normalized_loss = (
            args["normalized_loss"] if "normalized_loss" in args.keys() else False
        )
        self.continuous_features = (
            args["continuous_features"]
            if "continuous_features" in args.keys()
            else False
        )
        self.OHE_features = (
            args["OHE_features"] if "OHE_features" in args.keys() else False
        )
        if type(self.validation) not in (tuple, list, bool):
            raise TypeError(
                "!!VALIDATION must be a list or tuple containing the missing data, the original data, and the mask in this order"
            )

        self.cb = Callbacks(args)  # initialize Callbacks with same attributes

    def create_generator(self):
        """Initialize generator with two hidden Dense layers

        Returns:
            Keras model: Sequential Keras model 
        """
        model = Sequential(name="Generator")
        # accepts data and mask matrix as input, noise embedded in data
        model.add(Input(shape=(self.dim * 2,), name="Generator_input", dtype="float32"))
        model.add(Dense(units=self.dim, activation="relu", name="Gen_dense_1"))
        # model.add(Dropout(0.5)) #not part of original GAIN model either
        model.add(Dense(units=self.dim, activation="relu", name="Gen_dense_2"))
        # model.add(Dropout(0.5))
        model.add(Dense(units=self.dim, activation="sigmoid", name="Gen_output"))
        # model.summary()
        return model

    def create_discriminator(self):
        """Initialize Discriminator with two hidden Dense layers

        Returns:
            Keras model: Sequential Keras model
        """
        model = Sequential(name="Discrminator")
        # accepts data and hint tensor
        model.add(
            Input(shape=(self.dim * 2,), name="Discriminator_input", dtype="float32")
        )
        model.add(Dense(units=self.dim, activation="relu", name="Disc_dense_1"))
        # model.add(Dropout(0.5))
        model.add(Dense(units=self.dim, activation="relu", name="Disc_dense_2"))
        # model.add(Dropout(0.5))
        model.add(Dense(units=self.dim, activation="sigmoid", name="Disc_output"))
        # model.summary()
        return model

    @staticmethod
    def create_missing(data, miss_rate, replace_with=0.0):
        """generates missingness in a data. 
        Deprecated: Use MissingMechanism, especially for 
        data with OHE features.

        Args:
            data (numpy array): data to be 'made missing
            miss_rate (	float): between [0,1] proportion of missing data

        Returns:
            tuple: return missing data and the mask matrix
        """
        row, col = data.shape
        data_copy = data.copy()
        data_m = binary_sampler(1 - miss_rate, row, col)
        miss_data_x = data_copy.astype(np.float64)
        miss_data_x[
            data_m == 0
        ] = replace_with  # set to np.nan originally, introduces missingness

        return miss_data_x, data_m

    def prepare_train_pipeline(self, data, m_data):
        """Prepares missing data for training

        Args:
            data (np array): numpy array with 0. at missing values 
            m_data (np array{0,1}): mask indicating missingness where = 0

        Returns:
            tf.batchDataset: normalized data in
            tensorflow dataset with shuffled batches
        """
        # Initialize missingMechanism object
        self.missing_mechanism = MissingMechanism(self.args, data)
        self.row, self.dim = data.shape  # store dimensions
        ### Create C matrix to separate Continuous and binary vars
        self.C = create_C_total(self.OHE_features, self.batch_size, self.dim)

        # create C matrix to separete each OHE feature
        self.C_full = create_C_total(self.OHE_features, self.row, self.dim)
        if self.OHE_features:
            self.C_i_matrices = {}  # store C matrix for each feature
            for pair in self.OHE_features:
                self.C_i_matrices[pair] = create_C_i_per_feature(
                    pair, self.row, self.dim
                )  # key is column range, value is C matrix
        # scale data
        self.normalizer = MinMaxScaler()
        norm_data = self.normalizer.fit_transform(data)
        row, col = data.shape
        assert data.shape == norm_data.shape  # sanity check
        # Initialize generator and discriminator Neural Nets
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        # prepare data sets for batch distribution
        X_mb = norm_data
        M_mb = m_data
        # sample random vectors
        Z_mb = uniform_sampler(0, 0.01, row, col)
        # sample hint vector
        H_mb_temp = binary_sampler(self.hint, row, col)

        # combine with mask
        H_mb = M_mb * H_mb_temp

        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        # Create tensorflow batch dataset
        tensor_data = (
            tf.data.Dataset.from_tensor_slices(
                (X_mb.astype("float32"), M_mb.astype("float32"), H_mb.astype("float32"))
            )
            .shuffle(row)
            .batch(self.batch_size, drop_remainder=True)
        )

        return tensor_data

    # @tf.function
    def train_step(self, gen_opt, discr_opt, X, M, H):
        """One training step on one batch to optimize 
        Generator and Discriminator

        Args:
            gen_opt (tf.AdamOptimizer): Optimizer for Generator
            discr_opt (tf.AdamOptimizer): Optimizer for Discriminator
            (both optimizers initialized in .train method)
            X (tf.Dataset): one batch of training data
            M (tf.Dataset): corresponding mask to batch of training data
            H (tf.Dataset): corresponding hint matrix

        Returns:
            list: All losses calculated in the training step (used for logging)
        """

        X = tf.cast(X, dtype="float32")
        M = tf.cast(M, dtype="float32")
        H = tf.cast(H, dtype="float32")

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # imputed sample from batch
            G_sample = self.generator(tf.concat([X, M], axis=1))

            # combine with actual data using mask
            Hat_X = X * M + G_sample * (1 - M)

            # pass it to discriminator along with hint matrix
            D_prob = self.discriminator(tf.concat([Hat_X, H], axis=1))

            # calculate losses
            D_loss_temp = -tf.reduce_mean(
                (M * kb.log(D_prob + 1e-8) + (1 - M) * kb.log(1.0 - D_prob + 1e-8))
            )
            G_loss_temp = -tf.reduce_mean(
                (1 - M)
                * (
                    (1 - self.C) * kb.log(D_prob + 1e-8)
                    + self.C * X * kb.log(D_prob + 1e-8)  # new part of loss
                )
            )

            # extra loss for reconstruction
            MSE_loss = tf.math.reduce_sum(
                ((M * X - M * G_sample) ** 2) * (1 - self.C)
            ) / tf.math.reduce_sum(M * (1 - self.C))
            entropy_loss = tf.math.reduce_sum(
                self.C * (M * -X * kb.log(G_sample + 1e-8))
            ) / (
                tf.math.reduce_sum(M * self.C * X) + 1e-8
            )  # added X for correct mean reduction
            # finalize losses
            D_loss = D_loss_temp
            G_loss = (
                G_loss_temp + self.alpha * MSE_loss + self.beta * entropy_loss
            )  # alpha, beta are hyperparameters

        gradients_gen = gen_tape.gradient(G_loss, self.generator.trainable_variables)
        gradients_discr = disc_tape.gradient(
            D_loss, self.discriminator.trainable_variables
        )
        gen_opt.apply_gradients(zip(gradients_gen, self.generator.trainable_variables))
        discr_opt.apply_gradients(
            zip(gradients_discr, self.discriminator.trainable_variables)
        )
        # breakpoint()
        return G_loss, D_loss, G_sample, MSE_loss, entropy_loss

    def train(self, data, mask, ori_data):
        """Handles the entire training process INCLUDING data preprocessing

        Args:
            data (np.array): Training data with missing values
            mask (np.array): Mask to indicate which values are missing (1 == present value)
            ori_data (np.array): Original data without missing (only used for loss calculation)

        Returns:
            np.array: After training the imputed data set
        """
        # init optimizers
        gen_opt = tf.keras.optimizers.Adam(self.lr, name="GenOpt")
        discr_opt = tf.keras.optimizers.Adam(self.lr, name="DiscrOpt")
        # create tensor dataset
        tf_data = self.prepare_train_pipeline(data, mask)
        # calculate number of batches
        nr_batches = data.shape[0] / self.batch_size
        if self.normalized_loss:  # if normalized losses also logged
            ### normalize original data for normalized metrics
            ori_data_norm, _ = self._mask_normalize(ori_data)
            if self.validation:  # similarly for validation set if present
                test_ori_norm, _ = self._mask_normalize(self.validation[1])
            print("### Saving NORMALIZED metrics ###")

        print("Shhh, machine is learning...")
        previous_sample = np.zeros(10)  # used for debug if collapse
        savepoint = "MODELSAVE DISABLED"  # initialize variable
        last_generator = tf.keras.models.clone_model(
            self.generator
        )  # clone initial state
        rmse_ids = self.create_rmse_idxs(
            ori_data, self.OHE_features, self.missing_mechanism.num_cols
        )  # calculate RMSE ids (used for RMSE calculation in OHE features)

        for iter in tqdm(range(self.epochs)):  # start of optimization process
            # stores all losses from batch
            gen_runavg = []
            discr_runavg = []
            mse_loss_runavg = []
            entropy_loss_runavg = []
            batch_count = 0  # count current batch

            for X, M, H in tf_data:  # iterate over all batches in epoch
                gl, dl, G_sample, mse, entropy = self.train_step(
                    gen_opt, discr_opt, X, M, H
                )
                # check for collapse (doesn't happen anymore, here for safety)
                if (
                    np.sum(1 - np.isnan(G_sample)) == 0
                ):  # check if returns NaN ie. collapsed
                    print("COLLAPSE...")
                    break
                else:
                    previous_sample = G_sample
                # append losses
                gen_runavg.append(gl)
                discr_runavg.append(dl)
                mse_loss_runavg.append(mse)
                entropy_loss_runavg.append(entropy)
                batch_count += 1
                if batch_count / nr_batches > self.cb.batch_proportion():
                    # callback returns float [0,1] to only use proportion of batches if enabled
                    break  # go to next epoch

            # callback saves model if specified
            self.cb.save_model(iter, self.generator, self.discriminator)
            if (
                np.sum(1 - np.isnan(G_sample)) == 0
            ):  # IF COLLAPSE, cancel training process
                print("RETURNING LAST GENERATED SAMPLE...")
                print("RETURNING STABLE MODEL FROM EPOCH {}".format(savepoint))
                # callback creates log from all variables, saved in /log/ directory
                self.cb.create_log()
                return previous_sample, last_generator

            # calculate mean loss of epoch
            gen_avg_loss_epoch = tf.reduce_mean(gen_runavg).numpy()
            discr_avg_loss_epoch = tf.reduce_mean(discr_runavg).numpy()
            mse_avg_loss_epoch = tf.reduce_mean(mse_loss_runavg).numpy()
            entrop_avg_loss_epoch = tf.reduce_mean(entropy_loss_runavg).numpy()

            # add all losses to callback
            self.cb.add_log("Generator", gen_avg_loss_epoch)
            self.cb.add_log("Discriminator", discr_avg_loss_epoch)
            self.cb.add_log("Sum", gen_avg_loss_epoch + discr_avg_loss_epoch)
            self.cb.add_log("MSE Reconstruction", mse_avg_loss_epoch)
            self.cb.add_log("Entropy Reconstruction", entrop_avg_loss_epoch)
            pred = self.impute(data, mask)
            ### unnormalized RMSE always saved

            rmse = self.RMSE(ori_data, pred, mask, rmse_ids)
            self.cb.add_log("RMSE", rmse)

            ### normalize prediction if normalized loss enabled
            if self.normalized_loss:
                pred_norm = self.normalizer.transform(pred)
                rmse = self.RMSE(ori_data_norm, pred_norm, mask, rmse_ids)

                # log normalized losses
                self.cb.add_log("NORM__RMSE", rmse)

            ### log metrics on validation set if enabled
            if self.validation:

                # unnormalized RMSE
                test_imputed = self.impute(self.validation[0], self.validation[2])
                RMSE = self.RMSE(test_imputed, self.validation[1], self.validation[2])
                self.cb.add_log("RMSE Validation", RMSE)
                # if enabled also save normalized metric
                if self.normalized_loss:
                    test_imputed_norm, _ = self._mask_normalize(test_imputed)
                    RMSE = self.RMSE(
                        test_imputed_norm, test_ori_norm, self.validation[2]
                    )
                    self.cb.add_log("NORM__RMSE Validation", RMSE)

            # early stop if enabled (based on sum of gen and discr losses)
            early_stop = self.cb.early(gen_avg_loss_epoch + discr_avg_loss_epoch)

            if early_stop:  # if early stop returns True end training
                final_pred = self.impute(data, mask)
                self.cb.create_log()
                print("### EARLY STOPPING AT {} epoch".format(iter))
                return final_pred
        ### If training process finished without anomalies, create logfile here
        self.cb.create_log()
        print("Congratulations, the machine is smart now")

        final_pred = self.impute(data, mask)  # impute data
        return final_pred

    def ohe_argmax(self, gen_sample, new_C=None):
        """Transforms the array of probabilities onto OHE features
        by setting highest value to 1, rest to 0

        Args:
            gen_sample (tf.tensor): Sample coming from the Generator
            new_C (bool, optional): Whether generator sample is same as for training.
            Defaults to None.

        Returns:
            tf.tensor: f=OHE formatted generator output
        """
        sample_copy = gen_sample.numpy().copy()  # copy of generator output
        if new_C:  # check gen_sample is same as training
            n, k = sample_copy.shape
            # create new C matrices
            C_full = create_C_total(self.OHE_features, n, k)
            nulled = sample_copy * (1 - C_full)  # reset all OHE values to null
            C_i_matrices = {
                r: create_C_i_per_feature(r, n, k) for r in self.OHE_features
            }  # generate OHE feature separation matrices
            for (
                _,
                C,
            ) in C_i_matrices.items():  # find largest value and replace with 1
                index_arr = np.argmax((C * gen_sample), axis=1)  # index array
                index_arr2 = np.expand_dims(
                    index_arr, axis=1
                )  # reshaped index array of highest values for OHE cols
                # replace highest values in OHE cols with 1
                np.put_along_axis(nulled, index_arr2, 1, axis=1)
                # breakpoint()
        else:  # if Generator is used on same shaped data as for training
            nulled = sample_copy * (1 - self.C_full)  # reset all OHE values to null
            for (
                _,
                C,
            ) in self.C_i_matrices.items():  # find largest value and replace with 1
                index_arr = np.argmax((C * gen_sample), axis=1)  # index array
                index_arr2 = np.expand_dims(
                    index_arr, axis=1
                )  # reshaped index array of highest values for OHE cols
                # replace highest values in OHE cols with 1
                np.put_along_axis(nulled, index_arr2, 1, axis=1)

        return nulled

    def impute(self, data, m=None):
        """Impute a dataset with missing values
        returns DE-NORMALIZED data 

        Args:
            data (np array): data containing np.nan where missing!
            m (np.array, Optional): if input data does not 
            contain NaN for missing values, use this mask
        Returns:
            dataframe: imputed dataframe
        """
        # normalize data and generate mask
        if m is None:
            norm_data, mask = self._mask_normalize(data)
        else:  # Ignore generated mask and use passed one
            norm_data, _ = self._mask_normalize(data)
            mask = m
        # Replace missing values with noise (prior imputation)
        full_data = mask * norm_data + (1 - mask) * uniform_sampler(
            0, 0.01, data.shape[0], norm_data.shape[1]
        )
        # sample from generator
        gen_sample = self.generator(tf.concat([full_data, mask], axis=1))
        # If OHE features present in data set. NOTE: feature indices have to match up
        if self.OHE_features:
            # transform columns with OHE features
            if gen_sample.shape[0] != self.row:
                nulled = self.ohe_argmax(gen_sample, True)
            else:
                nulled = self.ohe_argmax(gen_sample)
            # combine imputation with originally present values
            X_bar = mask * norm_data + (1 - mask) * nulled
        else:
            # combine imputation with originally present values
            X_bar = mask * norm_data + (1 - mask) * gen_sample
        assert data.shape == X_bar.shape  # check if imputed has same shape
        imputed = self._denormalize(X_bar)  # denormalize data prior returning

        return imputed

    def _make_pred(self, data, mask):
        """DEPRECATED prediction function for continuous data sets only
            USE .impute for predictions.
        Args:
            data (np.array): data to be imputed
            mask (np.array): mask indicating missing values

        Returns:
            np.array: imputed data set
        """
        data_norm = self.normalizer.transform(data)
        generated_data = self.generator(tf.concat([data_norm, mask], axis=1))
        X_bar = mask * data_norm + (1 - mask) * generated_data
        imputed_data = self.normalizer.inverse_transform(X_bar)
        if self.rounding:
            imputed_data = np.around(imputed_data)
        return imputed_data

    def _mask_normalize(self, data):
        """Creates mask and normalizes data

        Args:
            data (np array): data containing np.nan as missing values
        Returns:
            norm_data, mask: Return normalized data set and missingness indicator mask
        """
        mask = 1 - np.isnan(data)
        mask = tf.cast(mask, "float32")
        data_copy = data.copy()
        data_copy[mask == 0] = 0.0
        check_is_fitted(
            self.normalizer
        )  # check if MinMaxScaler already fitted to train data
        norm_data = self.normalizer.transform(data_copy)
        return norm_data, mask

    def _denormalize(self, norm_data):
        """Denormalize features to their usual range
            Using the same scaler as for the training set
            (no re-fitting)
        Args:
            norm_data (np.array): normalized data set

        Returns:
            np.array: denormalized dataset
        """
        check_is_fitted(self.normalizer)
        denormalized = self.normalizer.inverse_transform(norm_data)
        if self.rounding:  # if rounding enabled
            imputed = np.round_(denormalized)
            return imputed
        return denormalized

    @staticmethod
    def create_rmse_idxs(data, OHE_features, continuous_features):
        """Creates matrix where rows indicate the indices of columns where continuous or OHE=1 data present 
            (use for RMSE calculations in data sets with OHE features)

        Args:
            data (np.ndarray): data to evaluate (original data)
            OHE_features (list): list of pairs for OHE indices
            continuous_features (list): list of continuous feature indices

        Returns:
            tuple: two matrices to only evaluate rows necessary 
        """
        if OHE_features:
            rmse_cols, rmse_rows = [], []
            for d_i, da in enumerate(data):

                ohe_ids = [
                    OHE_features[i][0] + np.where(da[range(*OHE_features[i])])[0][0]
                    if len(np.where(da[range(*OHE_features[i])])[0]) == 1
                    else 0
                    for i in range(len(OHE_features))
                ]  # index of each OHE feature per row
                rmse_col = continuous_features + ohe_ids
                rmse_row = np.ones_like(rmse_col) * d_i
                rmse_cols.extend(rmse_col)
                rmse_rows.extend(rmse_row)

            rmse_ids = np.array(rmse_rows), np.array(rmse_cols)
            return rmse_ids
        else:
            return False

    @staticmethod
    def RMSE(data, imputed_data, mask, rmse_ids=None):
        if rmse_ids:  # if OHE features present, use generated rmse idxs
            # calculate RMSE
            nominator = np.sum(
                (
                    (1 - mask[rmse_ids]) * data[rmse_ids]
                    - (1 - mask[rmse_ids]) * imputed_data[rmse_ids]
                )
                ** 2
            )
            denominator = np.sum(1 - mask[rmse_ids])
        else:  # only continuous variables
            nominator = np.sum(((1 - mask) * data - (1 - mask) * imputed_data) ** 2)
            denominator = np.sum(1 - mask)
        rmse = np.sqrt(nominator / denominator)
        return rmse

    @staticmethod
    def MSE(data, imputed_data, mask):
        """Mean Squared Error (only used for continuous features)

        Args:
            data (np.array): original data
            imputed_data (np.array): imputed data
            mask (np.array): mask indicating missingness

        Returns:
            float: mean squared error
        """
        nominator = np.sum(((1 - mask) * data - (1 - mask) * imputed_data) ** 2)
        denominator = np.sum(1 - mask)
        mse = nominator / float(denominator)
        return mse
