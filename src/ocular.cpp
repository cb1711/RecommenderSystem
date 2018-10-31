/*
 * Training of OCuLaR model
 */

#include "ocular.h"
#include "lineSearch.h"
#include "gradient.h"
#include "mpi.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ctime>
#include "halfUtils.h"
#include <string.h>
#define num_it 60

void ocular(int numItem, int numUser, int *csr_item, int *users, int *csr_user, int *items, uint16_t **short_items,
            uint16_t **short_users, int *alloted_item, int *alloted_user, int count_item, int count_user,
            int *proc_item, int *proc_user, int *displ_item, int *displ_user, int rank, int grp_size)
{
	std::random_device rand_dev;
	std::default_random_engine generator(rand_dev());
	std::uniform_real_distribution<float> distribution(1.0, 10.0);

    for (int i = 0; i < count_item; i++) {
		for (int j = 0; j < CLUSTERS; j++)
			short_items[alloted_item[i]][j] = float2half(distribution(generator));
	}

	for (int i = 0; i < count_user; i++) {
		for (int j = 0; j < CLUSTERS; j++)
			short_users[alloted_user[i]][j] = float2half(distribution(generator));
	}

	for (int i = 0; i < grp_size; i++) {
		proc_item[i] *= CLUSTERS;
		proc_user[i] *= CLUSTERS;
		displ_item[i] *= CLUSTERS;
		displ_user[i] *= CLUSTERS;
	}
    MPI_Request item_req, user_req;
	MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(short_items[0][0]), proc_item, displ_item, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &item_req);
	MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(short_users[0][0]), proc_user, displ_user, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &user_req);

	float **gi = new float *[numItem];
    float **gu = new float *[numUser];
    for (int item = 0; item < numItem; item++)
		gi[item] = new float[CLUSTERS];
	
	for (int user = 0; user < numUser; user++)
		gu[user] = new float[CLUSTERS];
	
	float *sum_item = new float[CLUSTERS];
	float *sum_user = new float[CLUSTERS];
	for (int i = 0; i < CLUSTERS; i++) {
		sum_item[i] = 0;
		sum_user[i] = 0;
	}
    float **fi, **fu;
    float *item_data = new float[numItem * CLUSTERS];
    float *user_data = new float[numUser * CLUSTERS];

    fi = new float *[numItem];
    fu = new float *[numUser];

    for (int i = 0; i < numItem; i++) {
        fi[i] = &(item_data[i * CLUSTERS]);
    }
    for (int i = 0; i < numUser; i++) {
        fu[i] = &(user_data[i * CLUSTERS]);
    }
    MPI_Wait(&item_req, MPI_STATUS_IGNORE);
    
    half2floatv(fi[0], short_items[0], numItem*CLUSTERS);

    for (int iter = 0; iter < num_it; iter++) {
        double start = clock();
		gradient(fi, fu, alloted_item, count_item, count_user, numUser, csr_item, users, sum_user, gi, user_req, short_users);
		linesearch(fi, sum_user, fu, gi, count_item, alloted_item, numItem, csr_item, users);
        double as=clock();
        float2halfv(fi[0],short_items[0],numItem*CLUSTERS);
        double ae=clock();
        std::cout<<"Conversion at "<<(ae-as)/CLOCKS_PER_SEC<< std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(short_items[0][0]), proc_item, displ_item, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &item_req);

	    memset(sum_item, 0, CLUSTERS*sizeof(int));
		gradient(fu, fi, alloted_user, count_user, count_item, numItem, csr_user, items, sum_item, gu, item_req, short_items);
		linesearch(fu, sum_item, fi, gu, count_user, alloted_user, numUser, csr_user, items);
        float2halfv(fu[0],short_users[0],numUser*CLUSTERS);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(short_users[0][0]), proc_user, displ_user, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &user_req);

        memset(sum_user, 0, CLUSTERS*sizeof(int));
	    
        double end = clock();
        std::cout<<(end-start)/CLOCKS_PER_SEC<<std::endl;
    }
    MPI_Wait(&user_req, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < numItem; i++)
		delete[] gi[i];
	for (int i = 0; i < numUser; i++)
    delete[] gu[i];
    delete[] item_data;
    delete[] user_data;
    delete[] gi;
    delete[] gu;
	delete[] sum_item;
    delete[] sum_user;
}
