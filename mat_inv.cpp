#include <mkl.h>
#include <mkl_lapacke.h>
#include <iostream>

using namespace std;

void dgemiv(double* A, double* A_inv, int m, int n)
{
    lapack_int info_getrf, info_getri;
    lapack_int ipiv[m];

    double* a = (double*)malloc(m*n*sizeof(double));

    cblas_dcopy(m*n, A, 1, a, 1);

    info_getrf = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, n, ipiv);
    info_getri = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, n, ipiv);

    cblas_dcopy(m*n, a, 1, A_inv, 1);
}

void mat_ptr(double* A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << A[i*n+j] << ' ';
        }
        cout << '\n';
    }
    
}

int main(void)
{
    double A[9] = {0, -3, -2, 1, -4, -2, -3, 4, 1};
    double A_inv[9] = {0};

    cout << "A = \n";
    mat_ptr(A, 3, 3);
    dgemiv(A, A_inv, 3, 3);
    cout << "\nA_inv = \n";
    mat_ptr(A_inv, 3,3 );
    
}