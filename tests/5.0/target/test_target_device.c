//===--- test_target_device.c ----------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks the target construct with device clause where device-
// modifier is either ancestor or device_num. If no device_modifier is 
// present, the behavior is the same as if device_num were present.
//
////===---------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define N 1028

int test_target_device_ancestor() {

    int i, which_device;
    int a[N];
    int errors = 0; 

    for (int i = 0; i < N; i++) {
        a[i] = i;
    }

    #pragma omp target device(ancestor: 1) map(tofrom: a, which_device) 
    {
        for (int i = 0; i < N; i++) {
            a[i] = a[i] + 2;
        }
 
        which_device = omp_is_initial_device();
    }

    OMPVV_TEST_AND_SET_VERBOSE(errors, which_device != 1);
    OMPVV_ERROR_IF(which_device != 1, "Target region was executed on device. Due to ancestor device-modifier,"
                                         "this region should execute on host");

    return errors;

}

int test_target_device_device_num() {
    
    int i, which_device, host_device_num, first_device_num;
    int b[N];
    int errors = 0; 

    for (int i = 0; i < N; i++) {
        b[i] = i;
    }

    host_device_num = omp_get_device_num(); 

    
    OMPVV_TEST_AND_SET_VERBOSE(errors, omp_get_num_devices <= 0);
    OMPVV_ERROR_IF(omp_get_num_devices <= 0, "Test fails, there are no target devices available");         

    if (omp_get_num_devices() > 0) {
         
        first_device_num = host_device_num + 1;
   
        #pragma omp target device(device_num: first_device_num) map(tofrom: b, which_device) 
        {
            for (int i = 0; i < N; i++) {
                b[i] = b[i] + 2;
            }
 
            which_device = omp_is_initial_device();
        }
    }

    OMPVV_TEST_AND_SET_VERBOSE(errors, which_device != 0);
    OMPVV_ERROR_IF(which_device != 0, "Target region was executed on host. Due to device num device-modifier,"
                                         "this region should execute on specified target device");   

    return errors;

}

int main() {

    int errors = 0;
   
    OMPVV_TEST_OFFLOADING;

    OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_device_ancestor());
    OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_device_device_num());

    OMPVV_REPORT_AND_RETURN(errors);
}
