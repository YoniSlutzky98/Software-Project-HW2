import sys
import pandas as pd
import numpy as np

def main():
    # Call all functions below
    # if AssertionError is raised - handle it!
    try:
        K, max_iter, eps, file_path_1, file_path_2 = receive_input()
    except (AssertionError):
        print("Invalid Input")
        return
    try:
        obs_1 = read_file(file_path_1)
        obs_2 = read_file(file_path_2)
    except:
        print("An Error Has Occurred")
        return
    try:
        obs = combine_tables(obs_1, obs_2)
        obs.sort_values(obs.columns[0], inplace=True)
        obs = obs.set_index(obs.columns[0])
    except:
        print("An Error Has Occurred")
        return
    try:
        validate_input(K, max_iter, obs.shape[0])
    except (AssertionError):
        print("Invalid Input!")
        return
    try:
        initial_centroids = kmeanspp(obs, K)
        print([int(x) for x in initial_centroids.index.values])
    except:
        print("An Error Has Occurred")
        return
    try:
        N = obs.shape[0]
        dim = obs.shape[1]
        final_centroids = C_function(N, K, max_iter, dim, eps, initial_centroids, obs) # TBC
        assert final_centroids != None
        return
    except:
        print("An Error Has Ocurred")
        return


'''
The function check if the input is of the right length.
Then, the function checks whether K and max_iter (if provided) are valid integers,
and whether eps is a valid float.
'''
def receive_input():
    assert len(sys.argv) in (5, 6)
    try:
        K = int(sys.argv[1])
    except:
        assert 1 == 0
    max_iter = 300
    i = 1
    if len(sys.argv) == 6:
        i += 1
        try:
            max_iter = int(sys.argv[2])
        except:
            assert 2 == 1
    try:
        eps = float(sys.argv[i+1])
    except:
        assert 3 == 2
    file_path_1 = sys.argv[i+2]
    file_path_2 = sys.argv[i+3]
    return K, max_iter, eps, file_path_1, file_path_2

'''
The function reads the input file.
The function returns a float matrix whose elements are that of the input file.
'''
def read_file(file_path):
    return pd.read_csv(file_path, header=None)

'''
The function inner joins the two input files using the first columns as keys.
'''
def combine_tables(obs_1, obs_2):
    return pd.merge(obs_1, obs_2, how='inner', left_on=obs_1.columns[0],
    right_on=obs_2.columns[0])

'''
The function checks if K and max_iter are of valid values.
'''
def validate_input(K, max_iter, N):
    assert 1<K<N and max_iter > 0

def kmeanspp(obs, K):
    np.random.seed(0)
    n = obs.shape[0]
    rand_index = np.random.choice(range(n))
    centroids = obs.iloc[[rand_index]]
    for i in range(1, K):
        distances = np.array([find_closest_distance(obs.iloc[[j]], centroids) for j in range(n)])
        s = sum(distances)
        probs = distances / s
        rand_index = np.random.choice(range(n), p=probs)
        centroids = pd.concat([centroids, obs.iloc[[rand_index]]])
    return centroids

'''
The function finds the distance of the closest centroid to x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is the same.
The function uses the function square_euclidean_distance.
'''
def find_closest_distance(x, centroids):
    minimal_distance = square_euclidean_distance(x, centroids.iloc[[0]])
    for i in range(1, centroids.shape[0]): # Find the centroid that brings distance to minimum.
        curr_distance = square_euclidean_distance(x, centroids.iloc[[i]])
        if curr_distance < minimal_distance:
            minimal_distance = curr_distance
    return minimal_distance

'''
The function returns the square of the euclidean distance between the two float dataframes.
The function assumes the dimension of the dataframes is the same.
'''
def square_euclidean_distance(x, y):
    s = 0
    for i in range(x.shape[1]):
        s += (x.iat[0, i] - y.iat[0, i]) ** 2
    return s

if __name__ == "__main__":
    main()