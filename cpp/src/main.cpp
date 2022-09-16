#include <iostream>
#include <Eigen/Dense>
#include <lsqr.h>

using Eigen::MatrixXd;
//using Eigen::SparseMatrix<double>;
 
int main()
{
   int my_res = test(3, 4);
   std::cout << my_res << std::endl;
  //SparseMatrix m(2,2);
  //m(0,0) = 3.0;
  //m(1,0) = 2.5;
  //m(0,1) = -1;
  //m(1,1) = m(1,0) + m(0,1);
  //std::cout << m << std::endl;


  //LsqrResult =  lsqr_sparse(const SparseMatrix &A, const Vector &b,
  //                          const double damp = 0.0, const double atol = 1e-8,
  //                          const double btol = 1e-8, const double conlim = 1e8,
  //                          int iter_lim = -1);

}

   
