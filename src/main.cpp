#include <iostream>
#include <mpi.h>
#include "ocular.h"
#include "lineSearch.h"
#include <math.h>
#include <iomanip>
#include <fstream>
#define MASTER 0

using namespace std;

int main(int argc, char *argv[])
{
    int numProcs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ifstream inFile;
    int tempTransfer[] = {0, 0, 0};
    if (rank == MASTER) {
        inFile.open("data/data");
        if(!inFile) {
            cerr<<"Couldn't open file" << endl;
            exit(0);
        }
        inFile >> tempTransfer[0] >> tempTransfer[1] >> tempTransfer[2];
    }
    MPI_Bcast(tempTransfer, 3, MPI_INT, MASTER, MPI_COMM_WORLD);
    int numItems = tempTransfer[0];
    int numUsers = tempTransfer[1];
    int numRatings = tempTransfer[2];
    int *csr_users = new int[numUsers + 1];
    int *items = new int[numRatings];
    int *csr_items = new int[numItems + 1];
    int *users = new int[numRatings];
#pragma omp parallel {
#pragma omp for
    for (int i = 0; i < numUsers + 1; i++) {
        csr_users[i] = 0;
    }
#pragma omp for
    for (int i = 0; i < numItems + 1; i++) {
        csr_items[i] = 0;
    }
#pragma omp for
    for (int i = 0; i < numRatings; i++) {
        items[i] = users[i] = 0;
    }
}
    if (rank == MASTER) {
        for (int i = 0; i < numRatings; i++) {
            int item, user;
            inFile >> item >> user;
            user--;
            users[i] = user;
            csr_items[item]++;
        }
        for (int i = 1; i <= numItems; i++) {
            csr_items[i] += csr_items[i - 1];
        }
        
        for (int i = 0; i < numRatings; i++) {
            int user, item;
            inFile >> item >> user;
            item--;
            items[i] = item;
            csr_users[user]++;
        }
        for (int i = 1; i <= numUsers; i++) {
            csr_users[i] += csr_users[i - 1];
        }
    }
    //setup mpi
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(users, numRatings, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(items, numRatings, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(csr_users, numUsers + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(csr_items, numItems + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	
    int *all_items = new int[numItems];
    int *all_users = new int[numUsers];
    int process_items = numItems / numProcs + (numItems % numProcs > rank);
    int process_users = numUsers / numProcs + (numUsers % numProcs > rank);
    int *alloted_items = new int[process_items];
    int *alloted_users = new int[process_users];
    
    //Divide work
    int *sendcounts_item = new int[numProcs];
    int *displs_item = new int[numProcs];
    int *sendcounts_user = new int[numProcs];
    int *displs_user = new int[numProcs];
    
    for (int i = 0; i < numProcs; i++) {
        sendcounts_item[i] = numItems / numProcs + (numItems % numProcs > i);
        if (i == 0)
            displs_item[i] = 0;
        else
            displs_item[i] = displs_item[i - 1] + sendcounts_item[i - 1];
    }
    
    for (int i = 0; i < numProcs; i++) {
        sendcounts_user[i] = numUsers / numProcs + (numUsers % numProcs > i);
        if (i == 0)
            displs_user[i] = 0;
        else
            displs_user[i] = displs_user[i - 1] + sendcounts_user[i - 1];
    }

    for (int i = 0; i < numItems; i++) {
        all_items[i] = i;
    }

    for (int i = 0; i < numUsers; i++) {
        all_users[i] = i;
    }

    MPI_Scatterv(all_items, sendcounts_item, displs_item, MPI_INT, alloted_items, process_items, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatterv(all_users, sendcounts_user, displs_user, MPI_INT, alloted_users, process_users, MPI_INT, MASTER, MPI_COMM_WORLD);

    //Call ocular
    float **fi, **fu;
    float *item_data = new float[numItems * CLUSTERS];
    float *user_data = new float[numUsers * CLUSTERS];

    fi = new float *[numItems];
    fu = new float *[numUsers];
    for (int i = 0; i < numItems; i++) {
        fi[i] = &(item_data[i * CLUSTERS]);
    }
    for (int i = 0; i < numUsers; i++) {
        fu[i] = &(user_data[i * CLUSTERS]);
    }
    ocular(numItems, numUsers, csr_items, users, csr_users, items, fi, fu, alloted_items, alloted_users, process_items, process_users, sendcounts_item, sendcounts_user, displs_item, displs_user, rank, numProcs);
    if (rank == MASTER) {
        cout << "Printing fi\n";
        for (int i = 0; i < numItems; i++) {
            for (int j = 0; j < CLUSTERS; j++) {
                cout << fi[i][j] << " ";
            }
            cout << endl;
        }

        cout << "Printing fu\n";
        for (int i = 0; i < numUsers; i++) {
            for (int j = 0; j < CLUSTERS; j++) {
                cout << fu[i][j] << " ";
            }
            cout << endl;
        }

        int user_id, item_id;
        cout << "Enter the user and item" << endl;
        while (cin >> item_id >> user_id) {
            if (user_id < 1 or user_id > numUsers or item_id < 1 or item_id > numItems)
                break;
            user_id--;
            item_id--;
            float x = innerProduct(fi[item_id], fu[user_id], CLUSTERS);
            cout << fixed << setprecision(2) << 100 * (1 - pow(M_E, -x)) << endl;
        }
    }
    MPI_Finalize();
    return 0;
}
