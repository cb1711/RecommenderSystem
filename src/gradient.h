#ifndef _gradient_h
#define _gradient_h
#include "mpi.h"

void gradient(float **items, float **users, int start_index, int numItems, int numUser, int totalUser, int *csr_items, int *csr_users, float *user_sum, float **g, MPI_Request &mpi_req, uint16_t **short_users);

#endif
