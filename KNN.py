#!/usr/bin/python3

# KNN classifier written in pure python3 for ROB11 Course at ENSTA ParisTech

from math import sqrt
from collections import Counter

class KNeighborsClassifier:
    X = []
    y = []
    n_neighbors = 0
    
    def __init__(self, n_neighbors=3):
        """
            Initializes a new KNN classifier. 
            n_neighbors (int) : the number of neighbors to use for the classification.
        """
        assert (n_neighbors > 0), "Number of neighbors must be positive"
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
            Fits the model with new data.
            X: list of features vectors
            y: list of target classes
        """
        assert (len(X)==len(y)), "Feature and target vectors must have the same size !"
        self.X += X
        self.y += y

    def norm(self, X, Y):
        """
            Computes Euclidian norm between two vectors.
        """
        assert (len(X)==len(Y)), "Vectors must have the same dimension"
        value = 0
        for x, y in zip(X,Y):
            value += (x - y)**2
        return sqrt(value)

    def predict(self, X):
        """
            Predicts the classes of given list of vectors.
            X: list of feature vectors
        """
        # Finding nearest neighbors
        NN = []
        features = [{"vector": x, "index": i} for i, x in enumerate(self.X)]
        for test_vector in X:
            # getting the list of nearest neighbors
            nn = sorted(features, key= lambda feature: self.norm(test_vector, feature["vector"]))
            # converting into list of known classes
            nn_classes = [self.y[neighbor["index"]] for neighbor in nn]
            NN.append(nn_classes[0: self.n_neighbors])

        # computing target class 
        prediction = []
        for classes in NN:
            classes_count = Counter(classes)
            prediction.append(max(classes_count, key=classes_count.get))
        return prediction

    def test(self, X, y):
        """
            Computes and returns the model's accuracy on a given test set.
            X: list of feature vectors
            y: list of matching classes targets
        """
        pred = self.predict(X)
        result = [x for x, y in zip(pred, y) if x==y]
        return len(result)/len(y)

# unit tests
if __name__ == "__main__":
    knn = KNeighborsClassifier()
    
    # norm function
    assert (knn.norm([1,1],[1,1]) == 0)
    assert (knn.norm([1,0],[1,1]) == 1)
    assert (knn.norm([1,1],[0,0]) == sqrt(2))
    
    # Fitting
    X = [
        [0, 0],
        [1, 0],
        [2, 2],
        [3, 4],
    ]
    
    y = [1, 1, 2, 2 ]

    knn.fit(X, y)
    assert (knn.X == X)
    assert (knn.y == y)

    # Classification
    assert knn.predict([[1, 0.5], [2.5, 2]]) == [1, 2]

    # Test
    assert knn.test([[1, 0.5], [2.5, 2]], [1, 2]) == 1.0
