#include <omp.h>
#include "lineSearch.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <iostream>

using namespace std;

/*
 *Computes inner product of two given vectors A,B of size
*/
float innerProduct(float *A, float *B, int size) {
    float val = 0;
    for (int i = 0; i < size; i++) {
        val = val + A[i] * B[i];
    }
    return val;
}

void likelihood(float *Q, bool *selected, float *user_sum, float **items, float **users, int numItems,
                int *item_sparse_csr_r, int *user_sparse_csr_c, int *allotted, int totalItems, bool type) {
    //allotted contains items allotted to the node
    //numItems has items allotted to the node
    //totalItems has total number of items in the dataset
    for (int i = 0; i < numItems; i++) {
        if (selected[i]) {
            if (type) //true for array which map using allotted array
                Q[i] = innerProduct(items[allotted[i]], user_sum, CLUSTERS) +
                       LAMBDA * innerProduct(items[allotted[i]], items[allotted[i]], CLUSTERS);
            else
                Q[i] = innerProduct(items[i], user_sum, CLUSTERS) + LAMBDA * innerProduct(items[i], items[i], CLUSTERS);

            int start = item_sparse_csr_r[allotted[i]];
            int end = item_sparse_csr_r[allotted[i] + 1];
            for (int j = start; j < end; j++) {
                int uid = user_sparse_csr_c[j];
                float x;
                if (type)
                    x = innerProduct(items[allotted[i]], users[uid], CLUSTERS);
                else
                    x = innerProduct(items[i], users[uid], CLUSTERS);
                float y = Q[i];
                Q[i] = Q[i] - x - log(1 - exp(-x)); //Replace with efficient implementation of e^x
            }
        }
    }
}

void linesearch(float **items, float *user_sum, float **users, float **gradient, int numItems, int *allotted, int totalItems,
           int *item_sparse_csr_r, int *user_sparse_csr_c) {
    //numItems is number of items allotted to the node
    //totalItems is number of items in all
    //allotted contains items allotted to the node
    float **newItems, **tempItems;
    tempItems = new float *[numItems];
    newItems = new float *[numItems];
    int removed[omp_get_max_threads()];
    memset(removed, 0, sizeof removed);
    for (int i = 0; i < numItems; i++) {
        newItems[i] = new float[CLUSTERS];
        tempItems[i] = new float[CLUSTERS];
    }
    bool *active = new bool[numItems];
    memset(active, true, numItems * sizeof(bool));
    float *Q = new float[numItems];
    float *Q2 = new float[numItems];
    likelihood(Q, active, user_sum, items, users, numItems, item_sparse_csr_r, user_sparse_csr_c, allotted, totalItems,
               true);
    double alpha = 1;
    bool flag = true;

    while (flag) {
        #pragma omp parallel for
        for (int i = 0; i < numItems; i++) {
            if (active[i])
                for (int j = 0; j < CLUSTERS; j++) {
                    //assert(isnan(gradient[allotted[i]][j]));
                    newItems[i][j] =
                            (items[allotted[i]][j] - alpha * gradient[allotted[i]][j]) > 0.0 ? (items[allotted[i]][j] -
                                                                                                alpha *
                                                                                                gradient[allotted[i]][j])
                                                                                             : 0.0;
                }
        }
        likelihood(Q2, active, user_sum, newItems, users, numItems, item_sparse_csr_r, user_sparse_csr_c, allotted,
                   totalItems, false);
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < numItems; i++) {
                if (active[i])
                    for (int j = 0; j < CLUSTERS; j++)
                        tempItems[i][j] = newItems[i][j] - items[allotted[i]][j];
            }
            #pragma omp for
            for (int i = 0; i < numItems; i++) {
                if (active[i]) {
                    if (Q2[i] - Q[i] <= SIGMA * innerProduct(gradient[allotted[i]], tempItems[i], CLUSTERS)) {
                        active[i] = false;
                        removed[omp_get_thread_num()]++;
                    }
                }
            }
        }
        alpha = alpha * BETA;
        int sum = 0;
        for (int i = 0; i < omp_get_max_threads(); i++)
            sum += removed[i];
        if (sum == numItems)
            flag = false;
    }
    delete[] active;
    delete[] Q;
    delete[] Q2;
    for (int i = 0; i < numItems; i++) {
        for (int j = 0; j < CLUSTERS; j++) {
            items[allotted[i]][j] = newItems[i][j];
        }
    }

    for (int i = 0; i < numItems; i++) {
        delete[] newItems[i];
        delete[] tempItems[i];
    }
    delete[] newItems;
    delete[] tempItems;
}
