/*
 *	Training of OCuLaR model
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

#define num_it 50

void ocular(int numItem, int numUser, int *csr_item, int *users, int *csr_user, int *items, uint16_t **short_items,
            uint16_t **short_users, int *alloted_item, int *alloted_user, int count_item, int count_user,
            int *proc_item, int *proc_user, int *displ_item, int *displ_user, int rank, int grp_size)
{
	std::random_device rand_dev;
	std::default_random_engine generator(rand_dev());
	std::uniform_real_distribution<float> distribution(0.1, 100.0);
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
	for (int item = 0; item < numItem; item++) {
		gi[item] = new float[CLUSTERS];
	}
	float **gu = new float *[numUser];
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



    half2floatv(fi, short_items, numItem, CLUSTERS);

	for (int iter = 0; iter < num_it; iter++) {

        double start = clock();
		gradient(fi, fu, alloted_item, count_item, count_user, csr_item, users, sum_user, gi, user_req, short_users);
		linesearch(fi, sum_user, fu, gi, count_item, alloted_item, numItem, csr_item, users);
        float2halfv(fi,short_items,numItem, CLUSTERS);
		MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(short_items[0][0]), proc_item, displ_item, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &item_req);

		for (int i = 0; i < CLUSTERS; i++) {
			sum_item[i] = 0;
		}
	    
		gradient(fu, fi, alloted_user, count_user, count_item, csr_user, items, sum_item, gu, item_req, short_items);
		linesearch(fu, sum_item, fi, gu, count_user, alloted_user, numUser, csr_user, items);
        float2halfv(fu,short_users,numUser, CLUSTERS);
		MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(short_users[0][0]), proc_user, displ_user, MPI_UNSIGNED_SHORT, MPI_COMM_WORLD, &user_req);

		for (int i = 0; i < CLUSTERS; i++)
			sum_user[i] = 0;
	    
        double end = clock();
        std::cout << double(end-start)/CLOCKS_PER_SEC << std::endl;
	}
	for (int i = 0; i < numItem; i++)
		delete[] gi[i];

	for (int i = 0; i < numUser; i++)
		delete[] gu[i];

	delete[] gi;
	delete[] gu;
	delete[] sum_item;
	delete[] sum_user;
}
