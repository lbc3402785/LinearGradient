#include "test.h"
#include "lineargradient.h"
#include "preconditionedconjugategradient.h"
#include "LevenbergMarquardt.h"
#include <iostream>
#include <Eigen/Dense>
/**
 * ————————————————
 *  版权声明：本文为CSDN博主「xuezhisdc」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
 *  原文链接：https://blog.csdn.net/xuezhisdc/article/details/54631490
 */
Test::Test()
{

}

void Test::testLinearGradient()
{
    Eigen::SparseMatrix<float> A(20,20);
    A.insert(6,3)=120.5;
    A.insert(10,1)=29.5;
    A.insert(15,0)=3.7;
    A.insert(19,16)=56.3;
    A.insert(19,17)=26.1;
    Eigen::SparseMatrix<float> ATA=A.transpose()*A;
    Eigen::SparseMatrix<float> I(20,20);
    I.setIdentity();
    ATA+=0.1*I;
    Eigen::VectorXf b=Eigen::VectorXf::Zero(20);
    b(0,0)=10;b(1,0)=1;b(2,0)=2;b(3,0)=3;b(4,0)=4;b(5,0)=5;b(6,0)=6;b(7,0)=7;b(8,0)=8;b(9,0)=9;b(10,0)=10;b(11,0)=11;b(12,0)=12;b(12,0)=28;
    //std::cout<<b<<std::endl;
//std::cout<<ATA<<std::endl;
    // 求解
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> chol(ATA);  // 执行A的 Cholesky分解
    Eigen::VectorXf x = chol.solve(b);         // 使用A的Cholesky分解来求解等号右边的向量b
    std::cout<<x<<std::endl;
    std::cout<<"--------"<<std::endl;
    std::cout<<LinearGradient<float,ATA.Options>::solve(ATA,b)<<std::endl;
    std::cout<<"--------"<<std::endl;
    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float> pgc;
    Eigen::VectorXf x1(20);

    Eigen::Diagonal<Eigen::SparseMatrix<float>> d=ATA.diagonal();
    d(0,0);
    Eigen::SparseMatrix<float> P(20,20);
    for(int i=0;i<20;i++){
        P.insert(i,i)= 1/d(i);
    }
    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float>::Status status=pgc.solve(ATA,b,x1,&P);
    std::cout<<x1<<std::endl;
    std::cout<<status.info<<std::endl;
    std::cout<<status.numIterations<<std::endl;
}

void Test::testIllMatrix()
{
    Eigen::SparseMatrix<float> A(2,2);
    A.insert(0,0)=1;
    A.insert(0,1)=0.99;
    A.insert(1,0)=0.99;
    A.insert(1,1)=0.98;
    Eigen::VectorXf b=Eigen::VectorXf::Ones(2);
    std::cout<<"A:"<<A<<std::endl;
    std::cout<<"--------"<<std::endl;
    std::cout<<"b:"<<b<<std::endl;
    std::cout<<"--------"<<std::endl;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> chol(A);  // 执行A的 Cholesky分解
    Eigen::VectorXf x = chol.solve(b);         // 使用A的Cholesky分解来求解等号右边的向量b
    std::cout<<x<<std::endl;
    std::cout<<"--------"<<std::endl;
    std::cout<<LinearGradient<float,A.Options>::solve(A,b)<<std::endl;
    std::cout<<"--------"<<std::endl;
    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float> pgc;
    Eigen::VectorXf x1(2);
    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float>::Status status=pgc.solve(A,b,x1);
    std::cout<<x1<<std::endl;
//    std::cout<<status.info<<std::endl;
//    std::cout<<status.numIterations<<std::endl;

}
void calcJacobian(Eigen::VectorXd& param,Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> &jac)
{
    jac(0,0)=2*param(0)-4.0;//2x-4
    jac(0,1)=0;
    jac(1,0)=2*(param(0)*param(1)-3)*param(1);//2(xy-3)*y
    jac(1,1)=2*(param(0)*param(1)-3)*param(0);//2(xy-3)*x
}
void calcError(Eigen::VectorXd& param,Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> &err)
{
    //f(0)=(x-2)^2
    err(0,0)=(param(0)-2)*(param(0)-2);
    //f(1)=(xy-3)^2
    err(1,0)=(param(0)*param(1)-3)*(param(0)*param(1)-3);
}
void Test::testLM()
{
    int nparams=2;
    int nerrs=2;
    OptimizationFramework::LevenbergMarquardt<double> solver(nparams,nerrs,OptimizationFramework::LevenbergMarquardt<double>::TermCriteria());

    Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> err, jac;
    Eigen::VectorXd matParams;
    matParams=solver.param;
    err.resize(nerrs,1);
    err.setZero();
    jac.resize(nerrs,nparams);
    jac.setZero();

    int iter = 0;

    for(;;)
    {
        const Eigen::VectorXd* _param;
        Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>* _jac;
        Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>* _err;

        bool proceed = solver.update(_param, _jac, _err);

        matParams=*_param;

        if (!proceed || !_err)
            break;

        if (_jac)
        {
            calcJacobian(matParams,jac);
            *_jac=jac;
        }

        if (_err)
        {
            calcError(matParams,err);

            iter++;
            *_err=err;
        }
    }
    std::cout<<"matParams:"<<matParams<<std::endl;
    std::cout<<"err:"<<err<<std::endl;
}
