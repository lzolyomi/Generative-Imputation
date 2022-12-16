import numpy as np


class MissingMechanism:
    """Class used to introduce missingness to training data,
    in correspondence with OHE features and specified missing proportion,
    and generate hint matrix.
    """

    def __init__(self, dct_args, data) -> None:
        """Initialize class

        Args:
            dct_args (dict): same arguments as for GAIN model
            data (np.array): training data for GAIN (without missingness)
        """
        assert (
            type(data) == np.ndarray
        ), f"ERROR Type of data {type(data)} not numpy array"
        self.data = data.astype(np.float64).copy()  # create copy of train data
        self.row, self.col = data.shape
        self.missing_rate = dct_args["missing_rate"]
        self.hint = dct_args["hint_rate"]
        self.OHE_features = (
            dct_args["OHE_features"] if "OHE_features" in dct_args.keys() else False
        )
        self.sort_columns()

    def sort_columns(self):
        """Sort all columns in data based on continuous or categorical features
        store in .num_cols and .cat_cols attribute
        """
        self.num_cols = []  # store index of numerical cols
        if self.OHE_features:  # check if OHE categorical features availabe
            # list of lists, one for each categorical feature, containing indices of OHE
            self.cat_cols = [
                tuple(range(pair[0], pair[1])) for pair in self.OHE_features
            ]

        else:
            self.cat_cols = []  # no OHE feature columns
        for idx in range(self.data.shape[1]):
            if not any([idx in r for r in self.cat_cols]):
                # if index not contained in any OHE feature column
                self.num_cols.append(idx)
        print(
            f"### {len(self.num_cols)} CONTINUOUS and {len(self.cat_cols)} OHE Features DETECTED"
        )

    def _create_missingness_numerical(self):
        """Replaces specified proportion of numerical columns with np.nan
        returns missingness indicator mask
        """
        for col in self.num_cols:
            binary_vector = self.binary_vector(self.missing_rate, (self.row,))

            self.data[:, col][binary_vector == 1] = np.nan
        return 1 - np.isnan(self.data)

    def _create_missingness_categorical(self):
        """Generates missingness for categorical features
        with OHE encoding (remove all values in the OHE columns)

        Returns:
            np.array: missingness indicator mask matrix
        """
        if self.OHE_features:  # check if any OHE features
            for feature in self.OHE_features:  # itearte over OHE features
                arr_feature = self.data[
                    :, feature[0] : feature[1]
                ]  # subset of data with columns belonging to one feature
                sum_rows = np.sum(arr_feature, axis=1)
                sum_not_one = np.sum(sum_rows != 1)  # check if any row has less/more 1s
                if sum_not_one > 0:  # if feature has rows contain less/more 1s
                    print(
                        f"### WARNING, for feature {feature} there are {sum_not_one} erroneous rows ###"
                    )
                for j in range(arr_feature.shape[1]):
                    binary_vector = self.binary_vector(self.missing_rate, (self.row,))
                    # 1 where OHE feature present AND random vector = 1
                    missing_vector = arr_feature[:, j] * binary_vector
                    # introduce np.nan to rows
                    self.data[:, feature[0] : feature[1]][missing_vector == 1] = np.nan
        return 1 - np.isnan(self.data)

    def replace_nan(self, replace_with=0.0):
        """replaces np.nan in data with given value
            returns the final mask
        Args:
            replace_with (float/nan, optional): value to replace nans with. Defaults to 0..
        """
        mask = 1 - np.isnan(self.data)
        self.data[mask == 0] = replace_with
        return mask

    def missingness_pipeline(self):
        """Runs the entire missingness creation process
        for both categorical and continuous features

        Returns:
            tuple: returns data with missing values and final missingness indicator mask
        """
        cat_miss = self._create_missingness_categorical()
        num_miss = self._create_missingness_numerical()

        final_mask = self.replace_nan()
        return self.data, final_mask

    def create_hint_matrix(self):
        """Generate hint matrix for GAIN training process

        Returns:
            np.array: hint matrix (noisy version of mask matrix)
        """
        hint = np.ones(self.data.shape)  # placeholder for hint matrix
        for j in self.num_cols:
            # replace each cont. column with a binary vector
            hint[:, j] = self.binary_vector(self.hint, (self.row,))
        for r in self.cat_cols:
            length = r[-1] - r[0]  # range of OHE feature
            assert length > 0, f"ERROR Length of OHE columns negative"
            binary_vec = self.binary_vector(self.hint, (1, self.row))
            ### contain same binary vector for all columns of OHE feature
            hint[:, r[0] : r[-1]] = np.repeat(binary_vec, length, axis=0).T

        return hint

    @staticmethod
    def binary_vector(p, size):
        """return binary random vector

        Args:
            p (float): probability of 1
            size (tuple,float): shape of vector

        Returns:
            np.array: random vector
        """
        return np.random.choice([0.0, 1.0], size, replace=True, p=[1 - p, p],)

