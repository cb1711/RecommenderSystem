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

#define num_it 500

void ocular(int numItem, int numUser, int *csr_item, int *users, int *csr_user, int *items, float **fi, float **fu, int *alloted_item, int *alloted_user, int count_item, int count_user, int *proc_item, int *proc_user, int *displ_item, int *displ_user, int rank, int grp_size)
{
	std::random_device rand_dev;
	std::default_random_engine generator(rand_dev());
	std::uniform_real_distribution<float> distribution(0.1, 100000.0);
	for (int i = 0; i < count_item; i++)
	{
		for (int j = 0; j < CLUSTERS; j++)
		{
			fi[alloted_item[i]][j] = distribution(generator);
		}
	}

	for (int i = 0; i < count_user; i++)
	{
		for (int j = 0; j < CLUSTERS; j++)
		{
			fu[alloted_user[i]][j] = distribution(generator);
		}
	}

	for (int i = 0; i < grp_size; i++)
	{
		proc_item[i] *= CLUSTERS;
		proc_user[i] *= CLUSTERS;
		displ_item[i] *= CLUSTERS;
		displ_user[i] *= CLUSTERS;
	}

	MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(fi[0][0]), proc_item, displ_item, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(fu[0][0]), proc_user, displ_user, MPI_FLOAT, MPI_COMM_WORLD);

	float **gi = new float *[numItem];
	for (int item = 0; item < numItem; item++)
	{
		gi[item] = new float[CLUSTERS];
	}
	float **gu = new float *[numUser];
	for (int user = 0; user < numUser; user++)
	{
		gu[user] = new float[CLUSTERS];
	}
	float *sum_item = new float[CLUSTERS];
	float *sum_user = new float[CLUSTERS];
	for (int i = 0; i < CLUSTERS; i++)
	{
		sum_item[i] = 0;
		sum_user[i] = 0;
	}
	for (int i = 0; i < numItem; i++)
	{
		for (int j = 0; j < CLUSTERS; j++)
		{
			sum_item[j] += fi[i][j];
		}
	}

	for (int i = 0; i < numUser; i++)
	{
		for (int j = 0; j < CLUSTERS; j++)
		{
			sum_user[j] += fu[i][j];
		}
	}
	for (int iter = 0; iter < num_it; iter++)
	{

		gradient(fi, fu, alloted_item, count_item, csr_item, users, sum_user, gi);
		linesearch(fi, sum_user, fu, gi, count_item, alloted_item, numItem, csr_item, users);
		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(fi[0][0]), proc_item, displ_item, MPI_FLOAT, MPI_COMM_WORLD);

		for (int i = 0; i < CLUSTERS; i++)
		{
			sum_item[i] = 0;
		}
		for (int i = 0; i < numItem; i++)
		{
			for (int j = 0; j < CLUSTERS; j++)
			{
				sum_item[j] += fi[i][j];
			}
		}

		gradient(fu, fi, alloted_user, count_user, csr_user, items, sum_item, gu);
		linesearch(fu, sum_item, fi, gu, count_user, alloted_user, numUser, csr_user, items);
		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &(fu[0][0]), proc_user, displ_user, MPI_FLOAT, MPI_COMM_WORLD);

		for (int i = 0; i < CLUSTERS; i++)
		{
			sum_user[i] = 0;
		}
		for (int i = 0; i < numUser; i++)
		{
			for (int j = 0; j < CLUSTERS; j++)
			{
				sum_user[j] += fu[i][j];
			}
		}
	}
	for (int i = 0; i < numItem; i++)
	{
		delete[] gi[i];
	}
	for (int i = 0; i < numUser; i++)
	{
		delete[] gu[i];
	}
	delete[] gi;
	delete[] gu;
	delete[] sum_item;
	delete[] sum_user;
}
