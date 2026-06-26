"""
AI-ENG-201 HW2 — Part 3.4
Text feature extraction: Bag-of-Words and TF-IDF from scratch.
No sklearn.feature_extraction used anywhere in this file.
Author: Fatima Alibabayeva
"""

import numpy as np
from collections import Counter
import re


class BagOfWords:
    """Bag-of-Words vectorizer.

    Builds a vocabulary of the top `max_features` words by total frequency
    across the training corpus, then transforms documents into count vectors.

    API mirrors sklearn's CountVectorizer (fit / transform / fit_transform).
    """

    def __init__(self, max_features: int = 5000) -> None:
        """Initialize BagOfWords.

        Parameters
        ----------
        max_features : maximum vocabulary size (top words by frequency).
        """
        self.max_features = max_features
        self.vocabulary_ = {}   # word -> column index mapping

    def _tokenize(self, text: str) -> list:
        """Lowercase and extract alphabetic tokens from a document.

        Parameters
        ----------
        text : raw document string

        Returns
        -------
        list of str tokens
        """
        # kicik herfe cevirib, yalniz herflerden ibaret sozleri saxlayiriq
        return re.findall(r'[a-z]+', text.lower())

    def fit(self, documents: list) -> "BagOfWords":
        """Build vocabulary from training documents.

        Parameters
        ----------
        documents : list of str

        Returns
        -------
        self
        """
        # butun documentlerdeki sözleri sayiriq
        word_counts = Counter()
        for doc in documents:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)

        # ən çox istifade edilen top max_features sözü vocabulary kimi saxlayiriq
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        return self

    def transform(self, documents: list) -> np.ndarray:
        """Transform documents to count matrix.

        Parameters
        ----------
        documents : list of str

        Returns
        -------
        np.ndarray, shape (N_docs, |vocabulary|)  — integer count matrix
        """
        N = len(documents)
        V = len(self.vocabulary_)
        X = np.zeros((N, V), dtype=np.float64)

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    # vocabulary-deki indeksine count elave edirik
                    X[i, self.vocabulary_[token]] += 1

        return X

    def fit_transform(self, documents: list) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        documents : list of str

        Returns
        -------
        np.ndarray, shape (N_docs, |vocabulary|)
        """
        return self.fit(documents).transform(documents)


class TfidfTransformer:
    """TF-IDF transformer.

    Converts a count matrix (from BagOfWords) to TF-IDF weights using:
        tf-idf_{t,d} = f_{t,d} * idf_t
        idf_t = log(N / (1 + |{d : f_{t,d} > 0}|))

    where f_{t,d} is the raw count of term t in document d,
    and the denominator uses +1 smoothing to avoid division by zero.
    """

    def __init__(self) -> None:
        """Initialize; IDF weights are set after calling fit."""
        self.idf_ = None  # hər söz ucun IDF deyerleri, shape (V,)

    def fit(self, X: np.ndarray) -> "TfidfTransformer":
        """Estimate IDF weights from a count matrix.

        Parameters
        ----------
        X : np.ndarray, shape (N_docs, V)  — count matrix from BagOfWords

        Returns
        -------
        self
        """
        N = X.shape[0]  # document sayi

        # hər sözün neçə documentdə göründüyünü sayırıq (ft,d > 0 şərti)
        doc_freq = np.sum(X > 0, axis=0)  # shape: (V,)

        # idf formula: log(N / (1 + |{d: f_{t,d} > 0}|))
        # +1 smoothing sifira bolunməni önləyir
        self.idf_ = np.log(N / (1 + doc_freq))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply TF-IDF weighting to a count matrix.

        Parameters
        ----------
        X : np.ndarray, shape (N_docs, V)

        Returns
        -------
        np.ndarray, shape (N_docs, V)  — TF-IDF weighted matrix
        """
        # TF-IDF = TF * IDF
        # broadcasting ile hər sözun IDF deyerini butun document-lere vururuq
        # document-ler uzre loop YOX — matris broadcasting istifade edilir
        return X * self.idf_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray, shape (N_docs, V)

        Returns
        -------
        np.ndarray, shape (N_docs, V)
        """
        return self.fit(X).transform(X)
