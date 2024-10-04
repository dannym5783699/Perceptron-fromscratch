import numpy as np


class WidrowHoff:
    _LOWER_BOUND = -1.0
    _UPPER_BOUND = 1.0

    def __init__(
            self,
            n_features: int,
            lr: float = 0.01,
            tol: float = 1e-4,
            max_epochs=100,
            verbose: int = 0,
            useBias: bool = False
    ) -> None:
        """Widrow-Hoff (LMS) learning model with a single output.

        Args:
            n_features (int): Number of features in data.
            lr (float, optional): Learning rate. Defaults to 0.01.
            tol (float, optional): Tolerance. Defaults to 1e-4.
            max_epochs (int, optional): Max number of epochs when training. Defaults to 100.
            verbose (int, optional): Verbosity option to output progress. Defaults to 0.
        """
        self._n_features = n_features
        self._lr = lr
        self._tol = tol
        self._max_epochs = max_epochs
        self._verbose = verbose
        self._num_iterations = None
        rng = np.random.default_rng()
        self._W = rng.uniform(self._LOWER_BOUND, self._UPPER_BOUND, self._n_features)
        self._b = rng.uniform(self._LOWER_BOUND, self._UPPER_BOUND)
        self._useBias = useBias

    def _validate_input(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[1] != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {x.shape[1]}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Expected {x.shape[0]} samples, got y {y.shape[0]}")

    @property
    def num_iterations(self) -> int:
        """Returns the number of iterations taken to converge.

        Returns:
            int: Number of iterations.
        """
        if self._num_iterations is None:
            raise ValueError("Model has not been trained yet.")
        return self._num_iterations

    def forward(self, x) -> np.ndarray:
        """Computes forward output for given input using fitted parameters.

        Args:
            x : Input data. Shape of (n_samples, n_features).

        Returns:
            np.ndarray: Continuous output result (not discretized).
        """
        if self._useBias:
            y_hat = np.dot(x, self._W) + self._b
            return y_hat
        else:
            y_hat = np.dot(x, self._W)
        return y_hat

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the given data using the Widrow-Hoff learning rule.

        Args:
            x (np.ndarray): Training input data. Shape of (n_samples, n_features).
            y (np.ndarray): Target output data. Shape of (n_samples, 1).

        Raises:
            ValueError: If number of features in input data does not match expected number of features.
            ValueError: If number of samples in input data does not match number of samples in target data.
        """

        self._validate_input(x, y)

        n_samples = x.shape[0]

        if self._verbose > 0:
            print(f"Beginning training with {n_samples} samples.")
        for current_epoch in range(self._max_epochs):

            e_squared_total = 0.0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                if self._verbose == 2:
                    print("Data: ", x_i, y_i)
                    print("\tOG Params: ", self._W)
                y_hat = self.forward(x_i)
                if self._verbose == 2:
                    print("\tY Hat: ", y_hat)

                # Computing error
                e_i = y_i - y_hat

                # If e_i is an array, sum its squared values
                e_squared = np.sum(e_i ** 2)

                # Accumulate the scalar squared error
                e_squared_total += e_squared
                if self._verbose == 2:
                    print("\tError: ", e_i, e_squared)

                # Updating params
                error_m = np.dot(e_i, x_i.T)
                if self._verbose == 2:
                    print("\tError applied to X: ", error_m)

                w_change = np.dot(self._lr, error_m)
                if self._verbose == 2:
                    print("\tError with Tol: ", w_change)

                w_new = np.add(self._W, w_change)
                if self._verbose == 2:
                    print("\tNew Params: ", w_new)

                self._W = w_new
                if self._useBias:
                    self._b = self._b + self._lr * e_i

            e_squared_total /= n_samples

            # Check if error is below tolerance
            print(f"Mean squared error: {e_squared_total}")
            if e_squared_total < self._tol:
                if self._verbose > 0:
                    print(f"Converged in {current_epoch} iterations.")
                    print(f"Finished training with {n_samples} samples.")
                    print(f"Final weights: {self._W}")
                    return

        print(f"Finished training with {n_samples} samples.")
        print(f"Final weights: {self._W}")
