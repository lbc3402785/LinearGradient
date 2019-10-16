#include "test.h"
#include "lineargradient.h"
#include <iostream>
Test::Test()
{

}

void Test::testLinearGradient()
{
    Eigen::Matrix<float,5,5> A=Eigen::Matrix<float,5,5>::Random();
    Eigen::Matrix<float,5,5> ATA=A.transpose()*A+0.1*Eigen::Matrix<float,5,5>::Identity();
    Eigen::Matrix<float,5,1> b;
    b(0,0)=0;b(1,0)=1;b(2,0)=2;b(3,0)=3;b(4,0)=4;
    //std::cout<<b<<std::endl;
    std::cout<<ATA.colPivHouseholderQr().solve(b)<<std::endl;
    std::cout<<"--------"<<std::endl;
    std::cout<<ATA.inverse()*b<<std::endl;
    std::cout<<"--------"<<std::endl;
    std::cout<<LinearGradient<float,ATA.Options>::solve(ATA,b)<<std::endl;
}
