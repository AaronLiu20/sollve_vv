//===--- test_target_memcpy_async_depobj.c ----------------------------===//
//
//  Inspired from OpenMP 5.1 Examples Doc, 5.16.4 & 8.9
//  This test utilizes the omp_target_memcpy_async construct to
//  allocate memory on the device asynchronously. The construct
//  uses 'obj' for dependency, so that memory is only copied once
//  the variable listed in the depend clause is changed.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int errors, i;

int test_target_memcpy_async_depobj() {

    int host = omp_get_initial_device();
    int device = omp_get_default_device();
    
    int array_size8 = N;
    int array_size7 = (N * .875);
    int array_size6 = (N * .750);
    int array_size5 = (N * .635);
    int array_size4 = (N * .500);
    int array_size3 = (N * .375);
    int array_size2 = (N * .250);
    int array_size1 = (N * .125);

    double *host_memory8 = (double *)malloc( sizeof(double)*array_size8);
    double *host_memory7 = (double *)malloc( sizeof(double)*array_size7);
    double *host_memory6 = (double *)malloc( sizeof(double)*array_size6);
    double *host_memory5 = (double *)malloc( sizeof(double)*array_size5);
    double *host_memory4 = (double *)malloc( sizeof(double)*array_size4);
    double *host_memory3 = (double *)malloc( sizeof(double)*array_size3);
    double *host_memory2 = (double *)malloc( sizeof(double)*array_size2);
    double *host_memory1 = (double *)malloc( sizeof(double)*array_size1);

    double *device_memory8 = (double *)omp_target_alloc( sizeof(double)*array_size8, device);

    OMPVV_TEST_AND_SET_VERBOSE(errors, device_memory8 == NULL);

    for(int i = 0; i < array_size8; i++){
        host_memory8[i] = i;
    }
    omp_depend_t obj;
    #pragma omp depobj(obj) depend(inout: device_memory8)
    omp_depend_t obj_arr[1] = {obj};

    /* copy to device memory */
    omp_target_memcpy_async(device_memory8, host_memory8, sizeof(double)*array_size8,
                                0,          0,
                                device,     host,
                                1,          obj_arr);

    #pragma omp taskwait depend(depobj: obj)
    #pragma omp target is_device_ptr(device_memory8) device(device) depend(depobj: obj)
    {
        for(int i = 0; i < array_size8; i++){
            device_memory8[i] = device_memory8[i]*2; // initialize data
        }
    }
    /* copy to host memory */
    omp_target_memcpy_async(host_memory8, device_memory8, sizeof(double)*array_size8,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory7, device_memory8, sizeof(double)*array_size7,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory6, device_memory8, sizeof(double)*array_size6,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory5, device_memory8, sizeof(double)*array_size5,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory4, device_memory8, sizeof(double)*array_size4,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory3, device_memory8, sizeof(double)*array_size3,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory2, device_memory8, sizeof(double)*array_size2,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    omp_target_memcpy_async(host_memory1, device_memory8, sizeof(double)*array_size1,
                                0,          0,
                                host,       device,
                                1,          obj_arr);

    #pragma omp taskwait depend(depobj: obj)
    for(int i = 0; i < array_size8; i++){
        OMPVV_TEST_AND_SET(errors, host_memory8[i]!=i*2);
    }
    // free resources
    free(host_memory8);
    free(host_memory7);
    free(host_memory6);
    free(host_memory5);
    free(host_memory4);
    free(host_memory3);
    free(host_memory2);
    free(host_memory1);

    omp_target_free(device_memory8, device);
    #pragma omp depobj(obj) destroy
    return errors;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_memcpy_async_depobj() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
