import numpy as np


def add_engineered_features(X):
    X_new = X.copy()

    X_new["MedInc_squared"] = X_new["MedInc"] ** 2
    X_new["Rooms_per_person"] = X_new["AveRooms"] / X_new["AveOccup"]
    X_new["Lat_Long_interaction"] = X_new["Latitude"] * X_new["Longitude"]

    return X_new