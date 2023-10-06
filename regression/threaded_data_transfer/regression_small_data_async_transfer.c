//===--- regression_small_data_async_transfers.c ------------------------------===//
//
// OpenMP API Version 4.5
//
// this tests performs asynchrous data transfer from the host to the device. this
// test is mean to stress out the runtime by performing 4 asynchrous operations 
// at once
//
// Routine being tested
// omp_target_memcpy_async
//
// Author: Aaron Liu <olympus@udel.edu> Oct 2023
////===----------------------------------------------------------------------===//

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define number_of_threads 10000

int main() {
  int errors = 0;
  int host = omp_get_initial_device();
  int device = omp_get_default_device();

  // allocate space on host & target
  int array_size4 = N;
  int array_size3 = (N * .75);
  int array_size2 = (N * .50);
  int array_size1 = (N * .25);

  double *device_memory4 =
      (double *)omp_target_alloc(sizeof(double) * array_size4, device);

  double *host_memory4 = (double *)malloc(sizeof(double) * array_size4);
  double *host_memory3 = (double *)malloc(sizeof(double) * array_size3);
  double *host_memory2 = (double *)malloc(sizeof(double) * array_size2);
  double *host_memory1 = (double *)malloc(sizeof(double) * array_size1);

  for (int i; i < array_size4; i++) {
    host_memory4[i] = i;
    if (i < array_size3) {
      host_memory3[i] = i;
    }
    if (i < array_size2) {
      host_memory2[i] = i;
    }
    if (i < array_size1) {
      host_memory1[i] = i;
    }
  }

  /* copy to target */
  omp_target_memcpy_async(device_memory4, host_memory4,
                          sizeof(double) * array_size4, 0, 0, device, host, 0,
                          NULL);

#pragma omp target is_device_ptr(device_memory4) device(device)

#pragma omp teams distribute parallel for num_threads(number_of_threads)
  for (int i = 0; i < N; i++) {
    device_memory4[i] = i * 2; // initialize data on device
  }

  /* copy to host */
  omp_target_memcpy_async(host_memory4, device_memory4,
                          sizeof(double) * array_size4, 0, 0, host, device, 0,
                          NULL);
  omp_target_memcpy_async(host_memory3, device_memory4,
                          sizeof(double) * array_size3, 0, 0, host, device, 0,
                          NULL);
  omp_target_memcpy_async(host_memory2, device_memory4,
                          sizeof(double) * array_size2, 0, 0, host, device, 0,
                          NULL);
  omp_target_memcpy_async(host_memory1, device_memory4,
                          sizeof(double) * array_size1, 0, 0, host, device, 0,
                          NULL);

#pragma omp taskwait
  for (int i = 0; i < N; i++) {
    if (host_memory4[i] != i * 2) {
      errors += 1;
    }
    if (i < array_size3) {
      if (host_memory3[i] != i * 2) {
        errors += 1;
      }
    }
    if (i < array_size2) {
      if (host_memory2[i] != i * 2) {
        errors += 1;
      }
    }
    if (i < array_size1) {
      if (host_memory1[i] != i * 2) {
        errors += 1;
      }
    }
  }
  // free resources
  free(host_memory4);
  free(host_memory3);
  free(host_memory2);
  free(host_memory1);

  omp_target_free(device_memory4, device);
  return errors;
}