#include <mkl.h>
#include <mkl_lapacke.h>
#include <vector>
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

void dgeminv(vector<double>& A, vector<double>& A_inv, int m, int n)
{
    lapack_int info_getrf, info_getri;
    lapack_int ipiv[m];

    double* a = (double*)malloc(m*n*sizeof(double));

    cblas_dcopy(m*n, &*A.begin(), 1, a, 1);

    info_getrf = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, n, ipiv);
    info_getri = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a, n, ipiv);

    cblas_dcopy(m*n, a, 1, &*A_inv.begin(), 1);
}

void dgemm(double alpha, vector<double>& A, vector<double>& B, double beta, vector<double>& C, int m, int n, int k)
{
    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, &*A.begin(), k, &*B.begin(), n, beta, &*C.begin(), n);
}

void dgemv(double alpha, vector<double>& A, vector<double>& x, double beta, vector<double>& y, int m, int n)
{
    cblas_dgemv (CblasRowMajor, CblasNoTrans, m, n, alpha, &*A.begin(), n, &*x.begin(), 1, beta, &*y.begin(), 1);
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

void mat_ptr(vector<double>&A, int m, int n)
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

void vec_ptr(vector<double>& vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i] << ' ';
    }
    cout << '\n';
}

int main(void)
{
    double A[9] = {0, -3, -2, 1, -4, -2, -3, 4, 1};
    double A_inv[9] = {0};

    vector<double> A_vec (A, A+9);
    vector<double> A_inv_vec (A_inv, A_inv+9);


    // cout << "A = \n";
    // mat_ptr(A, 3, 3);
    // dgemiv(A, A_inv, 3, 3);
    // cout << "\nA_inv = \n";
    // mat_ptr(A_inv, 3,3 );

    cout << "A = \n";
    mat_ptr(A_vec, 3, 3);
    dgeminv(A_vec, A_inv_vec, 3, 3);
    cout << "\nA_inv = \n";
    mat_ptr(A_inv_vec, 3,3 );

    cout << endl;

    vector<double> C(9);

    dgemm(1.0, A_vec, A_inv_vec, 1.0, C, 3, 3, 3);

    cout << "A*A_inv = \n";
    mat_ptr(C, 3,3);

    vector<double> I {1, 2, 3};
    vector<double> y {1e6, 1e6, 1e6};

    dgemv(1.0, C, I, 0.0, y, 3, 3);

    vec_ptr(y);
    
}