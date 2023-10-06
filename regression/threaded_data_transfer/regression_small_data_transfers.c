//===--- regression_large_data_transfers.c ---------------------------------===//
//
// OpenMP API Version 4.5
//
// this is meant to test stress the runtime for the data transfers using 8 
// differently sized arrays 
//
// Author: Aaron Liu <olympus@udel.edu> Oct 2023
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define number_of_threads 10000

int main() {
  int error = 0;

  int arraySizeLarge = N;
  int arraySizeMedium = N / 2;
  int arraySizeSmall = N / 4;

  int dataLarge[arraySizeLarge];
  int emptyLarge[arraySizeLarge];

  int emptyMedium[arraySizeMedium];
  int dataMedium[arraySizeMedium];

  int dataSmall[arraySizeSmall];
  int emptySmall[arraySizeSmall];

  for (int x = 0; x < arraySizeLarge; x++) {
    dataLarge[x] = x;
    if (x < arraySizeMedium) {
      dataMedium[x] = x;
    }
    if (x < arraySizeSmall) {
      dataSmall[x] = x;
    }
  }

#pragma omp target parallel for num_threads(number_of_threads)                 \
    map(to : dataLarge[0 : arraySizeLarge], dataMedium[0 : arraySizeMedium],   \
            dataSmall[0 : arraySizeSmall])                                     \
    map(from : emptyLarge[0 : arraySizeLarge],                                 \
            emptyMedium[0 : arraySizeMedium], emptySmall[0 : arraySizeSmall])
  for (int x = 0; x < arraySizeLarge; x++) {
    emptyLarge[x] = dataLarge[x];
    if (x < arraySizeMedium) {
      emptyMedium[x] = dataMedium[x];
    }
    if (x < arraySizeSmall) {
      emptySmall[x] = dataSmall[x];
    }
  }
  for (int x = 0; x < arraySizeLarge; x++) {
    if (dataLarge[x] != emptyLarge[x]) {
      error += 1;
    }
    if (x < arraySizeMedium) {
      if (dataMedium[x] != emptyMedium[x]) {
        error += 1;
      }
    }
    if (x < arraySizeSmall) {
      if (dataSmall[x] != emptySmall[x]) {
        error += 1;
      }
    }
  }
  // printf("program created %d amount of errors\n", error);
  return error;
}