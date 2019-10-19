#include "test.h"
#include "lineargradient.h"
#include "preconditionedconjugategradient.h"
#include <iostream>
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
    Eigen::VectorXf b(20);
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
    PreConditionedConjugateGradient<Eigen::SparseMatrix<float>,Eigen::VectorXf,float>::Status status=pgc.solve(ATA,b,x1);
    std::cout<<x1<<std::endl;
    std::cout<<status.info<<std::endl;
    std::cout<<status.numIterations<<std::endl;
}

void Test::testSparseMatrix()
{


}
