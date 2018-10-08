#ifndef _ocular_h
#define _ocular_h

#include <stdint.h>

void
ocular(int numItem, int numUser, int *csr_item, int *users, int *csr_user, int *items, uint16_t **fi, uint16_t **fu,
       int *alloted_item, int *alloted_user, int count_item, int count_user, int *proc_item, int *proc_user,
       int *displ_item, int *displ_user, int rank, int grp_size);

#endif
