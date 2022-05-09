/*
Assignment 2 - Software Project
In this assignment we were asked to implement a k-means++ algorithm in python with a C extension.
provided here is the C extension implementation.
*/
#include <stdio.h>
#include <stdlib.h>

/*
The function returns the square of the euclidean distance between the two double arrays.
The function assumes the dimension of the arrays is dim.
*/
double euclid_dist_sq(double* x, double* y, int dim){
    double dist;
    int j;

    dist = 0;
    for (j = 0; j < dim; j++){
        dist += (x[j]-y[j]) * (x[j]-y[j]);
    }
    return dist;
}

/*
The function finds the index of the closest centroid to the array x.
The distance is measured using the euclidean distance.
The function assumes the dimension of the centroids and of x is dim.
The function uses the function euclid_dist_sq.
*/
int find_closest(double** centroids, double* x, int K, int dim){
    double minimal_distance, curr_distance;
    int minimal_index, i;
    
    curr_distance = minimal_distance = euclid_dist_sq(centroids[0], x, dim);
    minimal_index = 0;
    for (i = 1; i < K; i++){
         curr_distance = euclid_dist_sq(centroids[i], x, dim);
         if (minimal_distance > curr_distance){
             minimal_distance = curr_distance;
             minimal_index = i;
         }
    }
    return minimal_index;
}

/*
This function is used to free memory allocated for 2D arrays.
The function assumes that the number of rows in the matrix is rows.
*/
void free_matrix(double** mat, int rows) {
    int i;

    for (i = 0; i < rows; i++)
    {
        free(mat[i]);
    }
    free(mat);
}

/*
The function calculates the K cluster centroids produced by the K-means algorithm on the observations.
The function receives the K points to serve as the centroids in the first iteration.
The function then iterates, performing the following:
- Adding each observation's elements to the sums of its closest cluster, and incrementing
  the updated size of the cluster. 
  Figuring out which is the closest cluster is done using the function find_closest.
- Calculating each cluster's new centroid as the average of the cluster's updated observations.
- Calculating each cluster's deviation between its old centroid and its new one. The deviation is calculated
using the function euclid_dist_sq.
The function stops when either max_iter iterations have happend, or when the deviation of any cluster
is less than epsilon squared (the distance itself is less than epsilon).
*/
int calculate_kmeans(double** obs, double** centroids, int N, int dim, int K, int max_iter, double epsilon){
    double** new_centroids;
    int* cluster_counts;
    int i, j, curr_index, converged;
    
    new_centroids = calloc(K, sizeof(double*));
    cluster_counts = calloc(K, sizeof(int));
    for (i = 0; i < K; i++){
        new_centroids[i] = calloc(dim, sizeof(double));
    }
    converged = 0;
    while (converged == 0 && max_iter > 0)
    {
        for (i = 0; i < N; i++){
            curr_index = find_closest(centroids, obs[i], K, dim);
            cluster_counts[curr_index] += 1;
            for (j = 0; j < dim; j++){
                new_centroids[curr_index][j] += obs[i][j];
            }
        }
        converged = 1;
        for (i = 0; i < K; i++){
            if (cluster_counts[i] == 0){
                free_matrix(new_centroids, K);
                free(cluster_counts);
                return 1;
            }
            for (j = 0; j < dim; j++){
                new_centroids[i][j] = new_centroids[i][j] / cluster_counts[i];
            }
            if (euclid_dist_sq(new_centroids[i], centroids[i], dim) >= (epsilon * epsilon)){
                converged = 0;
            }
        }
        max_iter--;
        for (i = 0; i < K; i++){
            cluster_counts[i] = 0;
            for (j = 0; j < dim; j++){
                centroids[i][j] = new_centroids[i][j];
                new_centroids[i][j] = 0;
            }
        }
    }
    free_matrix(new_centroids, K);
    free(cluster_counts);
    return 0;
}


int main(int argc, char const *argv[]) 
{
    int N, K, max_iter, dim;
    double eps;
    double** centroids;
    double** observations;

    if (calculate_kmeans(observations, centroids, N, dim, K, max_iter, eps) == 1)
    {
        printf("ERROR - handle in python");
        return 1;
    }
    else {
        printf("NO ERROR - handle in python");
    }
    
    return 0;
}
