#include "stdio.h"
#include "stdlib.h"

#define N 512

void add(int *a, int *b, int *c) {
    for (int idx=0; idx <N; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}

void arangeN(int *data) {
    for (int idx=0; idx<N; idx++) {
        data[idx] = idx;
    }
}

void print(int *a, int *b, int *c) {
    for (int idx=N-10; idx<N; idx++) {
        printf("\n %d + %d = %d", a[idx] , b[idx], c[idx]);
    }    
}

int main() {
    int *a, *b, *c;
    int size = N * sizeof(int);

    // allocate 
    a = (int *) malloc(size); 
    b = (int *) malloc(size); 
    c = (int *) malloc(size);

    // initialize
    arangeN(a); 
    arangeN(b);

    // perform operations
    add(a,b,c);
    print(a,b,c);
    free(a); free(b); free(c);
    return 0;
}