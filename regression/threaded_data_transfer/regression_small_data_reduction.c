// this tests uses 3 differently sized arrays to perform +=, -=, and *=
// reductions

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define number_of_threads 10000

int main() {
  int error = 0;

  int arraySizeLarge = N;
  int arraySizeMedium = N / 2;
  int arraySizeSmall = 20;

  int dataLarge[arraySizeLarge];
  int dataMedium[arraySizeMedium];
  int dataSmall[arraySizeSmall];
  int initialReductionValue = 1;

  int sumReduction = initialReductionValue;
  int minusReduction = initialReductionValue;
  int multiplyReduction = initialReductionValue;

  int testSum = initialReductionValue;
  int testMinus = initialReductionValue;
  int testMultiply = initialReductionValue;

  for (int x = 0; x < arraySizeLarge; x++) {
    dataLarge[x] = x;
    testSum += dataLarge[x];
    if (x < arraySizeMedium) {
      dataMedium[x] = x;
      testMinus -= dataMedium[x];
    }
    if (x < arraySizeSmall) {
      dataSmall[x] = x;
      if (x != 0) {
        testMultiply *= dataSmall[x];
      }
    }
  }

#pragma omp target parallel for num_threads(number_of_threads)                 \
    map(to : dataLarge[0 : arraySizeLarge], dataMedium[0 : arraySizeMedium],   \
            dataSmall[0 : arraySizeSmall]) reduction(+ : sumReduction)         \
    reduction(- : minusReduction) reduction(* : multiplyReduction)             \
    map(tofrom : sumReduction, minusReduction, multiplyReduction)
  for (int x = 0; x < arraySizeLarge; x++) {
    sumReduction += dataLarge[x];
    if (x < arraySizeMedium) {
      minusReduction -= dataMedium[x];
    }
    if (x < arraySizeSmall) {
      if (x != 0) {
        multiplyReduction *= dataSmall[x];
      }
    }
  }

  if (testSum != sumReduction) {
    error += 1;
  }
  if (testMinus != minusReduction) {
    error += 1;
  }

  if (testMultiply != multiplyReduction) {
    error += 1;
  }

  printf("program created %d amount of errors\n", error);
  printf(" testSum = %d sumReduction = %d\n testMinus = %d minusReduction = "
         "%d\n testMultiply = %d, multiplyReduction = %d\n",
         testSum, sumReduction, testMinus, minusReduction, testMultiply,
         multiplyReduction);
  return error;
}