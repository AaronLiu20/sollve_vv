// this tests uses 6 differently sized arrays and performs 3 data transfers
// within the pragma while also peforming the +=, -= and *= reductions

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512
#define number_of_threads 10000

int main() {
  int error = 0;

  int arraySizeLarge = N;
  int arraySizeMedium = N / 2;
  int arraySizeSmall = N / 4;
  int MultiplcationSize = 20;
  int initialReductionValue = 1;

  int dataLarge[arraySizeLarge];
  int testLarge[arraySizeLarge];

  int dataMedium[arraySizeMedium];
  int testMedium[arraySizeMedium];

  int dataSmall[arraySizeSmall];
  int testSmall[arraySizeSmall];

  int sumReduction = initialReductionValue;
  int minusReduction = initialReductionValue;
  int multiplyReduction = initialReductionValue;

  int testSum = initialReductionValue;
  int testMinus = initialReductionValue;
  int testMultiply = initialReductionValue;

  for (int x = 0; x < arraySizeLarge; x++) {
    dataLarge[x] = x;
    testSum += x;
    if (x < arraySizeMedium) {
      dataMedium[x] = x;
      testMinus -= dataMedium[x];
    }
    if (x < arraySizeSmall) {
      dataSmall[x] = x;
      if (x != 0 && x < MultiplcationSize) {
        testMultiply *= dataSmall[x];
      }
    }
  }

#pragma omp target parallel for num_threads(number_of_threads)                 \
    map(to : dataLarge[0 : arraySizeLarge], dataMedium[0 : arraySizeMedium],   \
            dataSmall[0 : arraySizeSmall]) reduction(+ : sumReduction)         \
    reduction(- : minusReduction) reduction(* : multiplyReduction)             \
    map(from : testSmall[0 : arraySizeSmall], testMedium[0 : arraySizeMedium], \
            testLarge[0 : arraySizeLarge])                                     \
    map(tofrom : sumReduction, minusReduction, multiplyReduction)
  for (int x = 0; x < arraySizeLarge; x++) {
    sumReduction += dataLarge[x];
    testLarge[x] = dataLarge[x];
    if (x < arraySizeMedium) {
      minusReduction -= dataMedium[x];
      testMedium[x] = dataMedium[x];
    }
    if (x < arraySizeSmall) {
      testSmall[x] = dataSmall[x];
      if (x != 0 && x < MultiplcationSize) {
        multiplyReduction *= dataSmall[x];
      }
    }
  }

  for (int x = 0; x < arraySizeLarge; x++) {
    if (dataLarge[x] != testLarge[x]) {
      error += 1;
    }
    if (x < arraySizeMedium) {
      if (dataMedium[x] != testMedium[x]) {
        error += 1;
      }
    }
    if (x < arraySizeSmall) {
      if (dataSmall[x] != testSmall[x]) {
        error += 1;
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