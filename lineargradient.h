#ifndef LINEARGRADIENT_H
#define LINEARGRADIENT_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
template<typename T,int _Options=Eigen::RowMajor>
class LinearGradient
{
public:
    LinearGradient();

    static Eigen::Matrix<T, Eigen::Dynamic,1, _Options> solve(const Eigen::SparseMatrix<T,_Options>& A,const Eigen::Matrix<T,Eigen::Dynamic, 1,_Options>& b,T epsilon=1e-6);
};
template<typename T, int _Options>
Eigen::Matrix<T, Eigen::Dynamic,1, _Options> LinearGradient<T,_Options>::solve(const Eigen::SparseMatrix<T,_Options> &A, const Eigen::Matrix<T, Eigen::Dynamic,1, _Options> &b,T epsilon)
{
    assert(A.rows()==A.cols());
    assert(A.cols()=b.rows());
    using namespace Eigen;
    typedef Eigen::Matrix<T, Eigen::Dynamic,1, _Options> VectorXT;
    int n=A.cols();
    VectorXT x0=VectorXT::Ones(n,1);
    VectorXT g0(n,1);
    g0=A*x0-b;
    VectorXT d0(n,1);
    d0=-g0;
    VectorXT xk(n,1);
    VectorXT gk(n,1);
    VectorXT dk(n,1);
    T alphak;
    T betak;
    gk=g0;
    xk=x0;
    dk=d0;
//    std::cout<<"g0:"<<gk<<std::endl;
//    std::cout<<"x0:"<<xk<<std::endl;
//    std::cout<<"d0:"<<dk<<std::endl;
    for(int i=1;i<=n;i++){
        if(gk.norm()<epsilon){
            std::cout<<"i:"<<i<<std::endl;
            break;
        }else{
            alphak=gk.squaredNorm()/(dk.dot(A*dk));
            VectorXT xkplus(n,1);
            xkplus=xk+alphak*dk;
            VectorXT gkplus(n,1);
            gkplus=A*xkplus-b;
            betak=gkplus.squaredNorm()/gk.squaredNorm();
            VectorXT dkplus(n,1);
            dkplus=-gkplus+betak*dk;
            //---------------------
            gk=gkplus;
            xk=xkplus;
            dk=dkplus;
        }
//        std::cout<<"gk:"<<gk<<std::endl;
//        std::cout<<"xk:"<<xk<<std::endl;
//        std::cout<<"dk:"<<dk<<std::endl;
    }
    return xk;
}
#endif // LINEARGRADIENT_H


