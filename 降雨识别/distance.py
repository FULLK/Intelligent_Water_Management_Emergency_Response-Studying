import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def calculate_euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def calculate_total_rainfall_distance(events):
    n = len(events)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_euclidean_distance(events[i], events[j])
    return distance_matrix

def calculate_center_similarity(center_a, center_b, total_points=450):
    overlap = len(set(center_a) & set(center_b))
    return overlap / total_points

def calculate_center_distance(events_centers):
    n = len(events_centers)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity = calculate_center_similarity(events_centers[i], events_centers[j])
            distance_matrix[i, j] = 1 - similarity
    return distance_matrix

def calculate_feature_distances(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    n = normalized_features.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = calculate_euclidean_distance(normalized_features[i], normalized_features[j])
    return distance_matrix