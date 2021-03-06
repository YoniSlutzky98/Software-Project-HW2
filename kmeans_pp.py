import sys
import pandas as pd
import numpy as np
import mykmeanssp

def main():
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
        obs = combine_tables(obs_1, obs_2) # Inner join the input files
        obs.sort_values(obs.columns[0], inplace=True) # Sort by the first column = indices
        original_indices = obs.iloc[:,0].to_list() # Extract original indices
        obs.drop(obs.columns[0], inplace=True, axis=1) # Remove indices from sorted merged df
        N = obs.shape[0]
        dim = obs.shape[1]
        obs = obs.to_numpy() # Convert df to np array for more efficient calculations.
    except:
        print("An Error Has Occurred")
        return
    try:
        validate_input(K, max_iter, N)
    except (AssertionError):
        print("Invalid Input!")
        return
    try:
        initial_centroids, initial_indices = kmeanspp(obs, K, N, original_indices)
        print(",".join(initial_indices))
    except:
        print("An Error Has Occurred")
        return
    try:
        final_centroids = mykmeanssp.fit(N, K, max_iter, dim, eps, 
        initial_centroids.tolist(), obs.tolist()) # Call C function
        assert final_centroids != None
        for centroid in final_centroids:
            print(",".join(["%.4f" % elem for elem in centroid]))
        return
    except:
        print("An Error Has Ocurred")
        return
    
'''
The function checks if the input is of the right length.
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
The function reads the input file as a dataframe.
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

'''
The function implements the kmeans++ algorithm.
The function uses weights to randomly choose the first K centroids from the observations.
The weights are assigned with regards to the euclidean distance from current centroids.
'''
def kmeanspp(obs, K, N, original_indices):
    np.random.seed(0)
    rand_index = np.random.choice(range(N))
    indices = [str(int(original_indices[rand_index]))] # Keeping track of original indices
    centroids = np.array([obs[rand_index]])
    for i in range(1, K):
        distances = np.array([find_closest_distance(obs[j], centroids) for j in range(N)])
        s = sum(distances)
        probs = distances / s
        rand_index = np.random.choice(range(N), p=probs)
        indices.append(str(int(original_indices[rand_index])))
        centroids = np.append(centroids, np.array([obs[rand_index]]), axis = 0)
    return centroids, indices

'''
The function finds the distance of the closest centroid to x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is the same.
'''
def find_closest_distance(x, centroids):
    minimal_distance = sum((x-centroids[0]) ** 2)
    for i in range(1, len(centroids)):
        minimal_distance = min(minimal_distance, sum((x-centroids[i]) ** 2))
    return minimal_distance

if __name__ == "__main__":
    main()