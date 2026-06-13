import numpy as np


class KNN:
    """
    k-Nearest Neighbors classifier / regressor implemented in vectorized NumPy.

    Parameters
    ----------
    k       : number of neighbors
    metric  : 'euclidean' | 'manhattan' | 'minkowski'
    q       : Minkowski exponent (used only when metric == 'minkowski')
    task    : 'classification' | 'regression'
    weights : 'uniform' | 'distance'  (distance weighting is the bonus part)
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",  # 'euclidean' | 'manhattan' | 'minkowski'
        q: float = 2.0,             # used only when metric == 'minkowski'
        task: str = "classification",  # 'classification' | 'regression'
        weights: str = "uniform",   # 'uniform' | 'distance' (bonus)
    ) -> None:
        self.k = k
        self.metric = metric
        self.q = q
        self.task = task
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """Memorize the training data. Returns self."""
        # KNN de train nodu deye just datani yadinda saxliyir
        self.X_train = X  # X → X_train
        self.y_train = y
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between every query point and every training point.
        Returns shape (n_queries, N_train) — fully vectorized, no Python loop over points.

        Memory: O(n_queries * N_train * p) for the diff tensor.
        Time  : O(n_queries * N_train * p) arithmetic operations.
        Broadcasting expands X from (n,1,p) and X_train from (1,N,p) into (n,N,p) diff.
        """
        # kokaltinda ferqlerin kvadratlari cemi
        diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]  # (n_q, N, p)
        if self.metric == "euclidean":
            distances = np.sqrt((diff ** 2).sum(axis=2))
        elif self.metric == "manhattan":  # 𝑑=|𝑥1−𝑥2|+|𝑦1−𝑦2|
            distances = np.abs(diff).sum(axis=2)
        elif self.metric == "minkowski":  # \(D(A,B)=(\sum|A_i-B_i|^p)^{1/p}\)
            distances = ((np.abs(diff) ** self.q).sum(axis=2)) ** (1.0 / self.q)
        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels (classification) or values (regression)."""
        # P.S burda bonusda var
        distances = self._compute_distances(X)
        # her setiri sortda ve ilk k sutunun indexini goturur
        k_ind = np.argsort(distances, axis=1)[:, : self.k].astype(int)
        k_label = self.y_train[k_ind]  # hemin indexlere uygun labellari gotur 0 or 1

        # ---- REGRESSION ----
        if self.task == "regression":
            if self.weights == "uniform":
                return k_label.mean(axis=1)
            elif self.weights == "distance":
                k_dist = np.array([distances[i, k_ind[i]] for i in range(len(X))])
                w = 1.0 / (k_dist + 1e-8)  # weightleri hesabla
                return (w * k_label).sum(axis=1) / w.sum(axis=1)

        # ---- CLASSIFICATION ----
        if self.weights == "uniform":
            # 0 nan 1leri sayir ve maxi goturur
            # .astype(int) lazimdi cunki bincount float kabul etmir
            predictions = np.array(
                [np.bincount(row.astype(int)).argmax() for row in k_label]
            )
        elif self.weights == "distance":
            k_dist = np.array([distances[i, k_ind[i]] for i in range(len(X))])
            """
            ozumcun yazmisamki basa dusum
            distances = [[0.1, 5.0, 0.2, 3.0],  # test 0in butun train-nere mesafesi
                         [2.0, 0.3, 4.0, 0.1]]   # test 1-in

             k_ind = [[0, 2, 3],   # test 0-in en yaxin 3nun indexi
                       [1, 3, 0]]   # test 1-in

            i=0: distances[0, [0,2,3]] → [0.1, 0.2, 3.0] — test 0-ın k qonşusuna məsafələr
            i=1: distances[1, [1,3,0]] → [0.3, 0.1, 2.0] — test 1-in k qonşusuna məsafələr
            result:
            k_dist = [[0.1, 0.2, 3.0],
            [0.3, 0.1, 2.0]]
            """
            # weightderni hesabla yeniki yaxinda olan daha cox tesir edir uzaqda olandan
            w = 1.0 / (k_dist + 1e-8)
            predictions = []
            for i in range(len(X)):
                w0 = w[i][k_label[i] == 0].sum()  # survived=0 olanların ağırlıq cəmi
                w1 = w[i][k_label[i] == 1].sum()  # survived=1 olanların ağırlıq cəmi
                predictions.append(0 if w0 >= w1 else 1)  # hansi coxdursa onu qaytar; beraberdirse uniform kimi 0 secirik
            predictions = np.array(predictions)
        else:
            raise ValueError(f"Unknown weights: {self.weights!r}")
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-class probabilities (classification only).
        Output shape: (n_queries, n_classes).

        Per Equation 1: p̂(y=c|x) = (1/k) * Σ_{i∈Nk(x)} 1{y^(i)=c}
        Probabilities live in {0, 1/k, 2/k, ..., 1} — only k+1 distinct values.
        """
        assert self.task == "classification", "predict_proba is only for classification"
        distances = self._compute_distances(X)
        # her setiri sortda ve ilk k sutunun indexini goturur
        k_ind = np.argsort(distances, axis=1)[:, : self.k].astype(int)
        k_label = self.y_train[k_ind]  # hemin indexlere uygun labellari gotur 0 or 1

        if self.weights == "uniform":  # bonus
            proba_1 = k_label.mean(axis=1)
        elif self.weights == "distance":  # yeniki heresi ferali distance dadi eyni qeder degilde
            k_dist = np.array([distances[i, k_ind[i]] for i in range(len(X))])
            w = 1.0 / (k_dist + 1e-8)
            proba_1 = (w * k_label).sum(axis=1) / w.sum(axis=1)
        else:
            raise ValueError(f"Unknown weights: {self.weights!r}")
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])


# ikinci bonusdu, hazir balltreenen
# demeli normalda biz bilirikii meselen 1000 point varsa hamsiicin mesafe hesablanacaq
# ama burda 500 sagdaki 500 solkdai olaraq ayriqiri ve bizim bu pointimiz hansi terefde
# olsa o terefden gotrururk
from sklearn.neighbors import BallTree


class ApproxKNN:
    """
    BallTree-based KNN wrapper with the same public API as KNN.

    This class is used only for the optional bonus. BallTree performs exact
    nearest-neighbor search for the supported metric, but the benchmark still
    shows the speed behavior as leaf_size changes on a large synthetic dataset.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        q: float = 2.0,
        task: str = "classification",
        weights: str = "uniform",
        leaf_size: int = 30,
    ) -> None:
        self.k = k
        self.metric = metric
        self.q = q
        self.task = task
        self.weights = weights
        self.leaf_size = leaf_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ApproxKNN":
        """Build the BallTree on training data. Returns self."""
        if self.metric == "euclidean":
            tree_metric = "euclidean"
        elif self.metric == "manhattan":
            tree_metric = "manhattan"
        elif self.metric == "minkowski":
            tree_metric = "minkowski"
        else:
            raise ValueError(f"Unknown metric for ApproxKNN: {self.metric!r}")
        self.y_train = y
        self.tree = BallTree(X, leaf_size=self.leaf_size, metric=tree_metric)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values using BallTree neighbor lookup."""
        distances, k_ind = self.tree.query(X, k=self.k)
        k_label = self.y_train[k_ind]

        if self.task == "regression":
            if self.weights == "uniform":
                return k_label.mean(axis=1)
            if self.weights == "distance":
                w = 1.0 / (distances + 1e-8)
                return (w * k_label).sum(axis=1) / w.sum(axis=1)
            raise ValueError(f"Unknown weights: {self.weights!r}")

        if self.weights == "uniform":
            return np.array([np.bincount(row.astype(int)).argmax() for row in k_label])
        if self.weights == "distance":
            w = 1.0 / (distances + 1e-8)
            predictions = []
            for i in range(len(X)):
                w0 = w[i][k_label[i] == 0].sum()
                w1 = w[i][k_label[i] == 1].sum()
                predictions.append(0 if w0 >= w1 else 1)
            return np.array(predictions)
        raise ValueError(f"Unknown weights: {self.weights!r}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probabilities for classification only."""
        assert self.task == "classification", "predict_proba is only for classification"
        distances, k_ind = self.tree.query(X, k=self.k)
        k_label = self.y_train[k_ind]
        if self.weights == "uniform":
            proba_1 = k_label.mean(axis=1)
        elif self.weights == "distance":
            w = 1.0 / (distances + 1e-8)
            proba_1 = (w * k_label).sum(axis=1) / w.sum(axis=1)
        else:
            raise ValueError(f"Unknown weights: {self.weights!r}")
        return np.column_stack([1 - proba_1, proba_1])
