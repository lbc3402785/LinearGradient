
#ifndef LEVENBERGMARQUARDT_H
#define LEVENBERGMARQUARDT_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include<Eigen/Jacobi>
#include <iostream>
namespace OptimizationFramework{
#define TERMCRIT_ITER   1
#define TERMCRIT_EPS 2

/**
 *  为了便于实现和算法效率，要求不改变的参数放在前半部分或者后半部分连续存放
 */
template<typename T,int _Options=Eigen::RowMajor>
class LevenbergMarquardt
{
public:
    bool change;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic,Eigen::Dynamic, _Options> MatT;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic,1, Eigen::ColMajor> VectorXT;
    enum Type{
        COUNT =1,
        MAX_ITER =COUNT,
        EPS =2
    };
    enum DecompTypes {
        /** Gaussian elimination with the optimal pivot element chosen. */
        DENSE_LU       = 0,
        SPARSE_LU       = 1,
        /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
        src1 can be singular */
        DECOMP_SVD      = 2,
        /** eigenvalue decomposition; the matrix src1 must be symmetrical */
        DECOMP_EIG      =3,
        /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
        defined */
        DENSE_CHOLESKY = 4,
        SPARSE_CHOLESKY = 5,
        /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
        DENSE_QR       = 6,
        SPARSE_QR       = 7,
        /** while all the previous flags are mutually exclusive, this flag can be used together with
        any of the previous; it means that the normal equations
        \f$\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}\f$ are
        solved instead of the original system
        \f$\texttt{src1}\cdot\texttt{dst}=\texttt{src2}\f$ */
        DECOMP_NORMAL   = 16
    };
    enum { DONE=0, STARTED=1, CALC_J=2, CHECK_ERR=3 };
    struct TermCriteria{
        TermCriteria(){
            type=TERMCRIT_EPS;
            epsilon=1e-6;
            max_iter=30;
        }
        TermCriteria(int type,int max_iter,T epsilon):type(type),epsilon(epsilon),max_iter(max_iter){

        }

        int type;
        T epsilon;
        int max_iter;
    };
    LevenbergMarquardt();
    LevenbergMarquardt( int nparams, int nerrs, TermCriteria criteria=
            TermCriteria(TERMCRIT_EPS+TERMCRIT_ITER,5000000,DBL_EPSILON),
                        bool completeSymmFlag=false );
    ~LevenbergMarquardt(){}
    std::vector<unsigned char> mask;              // 标记优化过程中不优化的量。0代表不优化，1代表优化。大小跟param相同
    VectorXT prevParam;         // 前一步的优化参数，用途有两个：1、判断变量是否在变化，不变化停止迭代。2、在此参数上减去一个高斯方向得到下一个参数。
    VectorXT param;             // 所有要优化的变量集合
    MatT J;                 // 目标函数的偏导数
    MatT err;               // 测量误差向量
    MatT JtJ;               // 正规方程左侧的部分
    MatT JtJN;              // 去掉非优化变量后的正规方程左侧部分
    MatT JtErr;             // 正规方程右侧的部分
    MatT JtJV;              // 去掉非优化变量的JtErr，
    VectorXT JtJW;              // 去掉非优化变量后待求向量
    T prevErrNorm,errNorm;  // 测量误差，更新前的测量误差，更新后的测量误差
    int lambdaLg10;         // LM 算法中的lanbda，此处的lamdaLg10 加1代表lambda变大10倍，减1缩小10倍
    TermCriteria criteria;
    int state;              // 优化步骤中的状态
    int iters;              // 记录迭代的次数
    bool completeSymmFlag;  // 使去掉非优化变量后的JtJN变成对称矩阵，只是在updateAlt函数是有用。
    int solveMethod;        // 正规方程的解法
    void init( int nparams, int nerrs, TermCriteria criteria=TermCriteria(TERMCRIT_EPS+TERMCRIT_ITER,10000,DBL_EPSILON),
               bool completeSymmFlag=false );
    bool update(const VectorXT*& param, MatT*& J, MatT*& err);
    bool updateAlt(const VectorXT*& param, MatT*& JtJ, MatT*& JtErr, T& errNorm );
    void step();
    void completeSymm( MatT& m, bool LtoR);
    bool solve(MatT&JtJN, MatT&JtJV, VectorXT&JtJW, int& solveMethod);
    static void subMatrix(const MatT& src, MatT& dst, const std::vector<unsigned char>& cols,
                          const std::vector<unsigned char>& rows);
private:
    void clear();
    T det2(MatT& m){
        return (m(0,0)*m(1,1)-m(0,1)*m(1,0));
    }
    T det3(MatT&m){
        return (m(0,0)*(m(1,1)*m(2,2) - m(1,2)*m(2,1)) -
                m(0,1)*(m(1,0)*m(2,2) - m(1,2)*m(2,0)) +
                m(0,2)*(m(1,0)*m(2,1) - m(1,1)*m(2,0)));
    }
};



template<typename T, int _Options>
LevenbergMarquardt<T,_Options>::LevenbergMarquardt()
{
    lambdaLg10 = 0; state = DONE;
    criteria = TermCriteria(0,0,0);
    iters = 0;
    completeSymmFlag = false;
    errNorm = prevErrNorm = DBL_MAX;
    solveMethod = DECOMP_SVD;
}

template<typename T, int _Options>
LevenbergMarquardt<T,_Options>::LevenbergMarquardt(int nparams, int nerrs, LevenbergMarquardt::TermCriteria criteria, bool completeSymmFlag)
{
    init(nparams, nerrs, criteria, completeSymmFlag);
}

/*
    * 功能：
    * 参数：
    *   [in]      nparams                    优化变量的参数
    *   [in]      nerrs                      nerrs代表样本的个数（一个测量点代表两个样本x和y）。
                                             nerrs = 0，优化时使用updateAlt函数，因为此函数不关心样本个数（JtJ外部计算，自然不需要知道J的大小）
                                             nerrs > 0，优化时使用update函数，应为内部需要使用J的大小，所以J的维度需要提前声明，其维度大小为nerrs*nparams
    *   [in]      criteria                   迭代停止标准
    *   [in]      completeSymmFlag           当nerrs=0时有效。防止外部计算得到的JtJ不对称。此标记位不管是false和true都会使JtJ 变为对称。
    *
    *  返回值：
    *
    *  备注：
    *             此函数相对于updateAlt函数，计算相对简单，但自己自由发挥的空间较少。例如代权重的最小二乘此函数就无法实现。
    *
    */
//————————————————
//版权声明：本文为CSDN博主「xuelangwin」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/xuelangwin/article/details/81095651
template<typename T, int _Options>
void LevenbergMarquardt<T,_Options>::init(int nparams, int nerrs, LevenbergMarquardt::TermCriteria criteria0, bool completeSymmFlag)
{
    //if(nerrs=1)nerrs=2;
    //    if(param.rows() != nparams || nerrs != err.rows())
    //        clear();
    change=false;
    mask.assign(nparams, 1);
    prevParam.setZero(nparams, 1);
    param.setZero(nparams,1);
    JtJ.setZero(nparams, nparams);
    JtErr.setZero(nparams, 1);
    if( nerrs > 0 )
    {
        J.setZero(nerrs, nparams);
        err.setZero(nerrs, 1);
    }
    errNorm = prevErrNorm = DBL_MAX;
    lambdaLg10 = -3;
    this->criteria = criteria0;

    if( criteria.type & TERMCRIT_ITER ){
        criteria.max_iter = std::min(std::max(criteria.max_iter,1),10000000);
    }
    else{
        criteria.max_iter = 30;
    }
    if( criteria.type & TERMCRIT_EPS )
        criteria.epsilon = std::fmax(criteria.epsilon, 0);
    else
        criteria.epsilon = DBL_EPSILON;
    state = STARTED;
    iters = 0;
    completeSymmFlag = completeSymmFlag;
    solveMethod = DECOMP_SVD;
}
/*
* 功能：
* 参数：
*   [out]      param       需要优化的参数，根据内部参数的优化状况，对当前参数进行输出，方便外部误差的计算
*   [in]       J           目标函数的偏导数，内部计算JtJ
*   [in]       err         内部计算JtErr 和 errNorm
*
*  返回值：
*
*  备注：
*             此函数相对于updateAlt函数，计算相对简单，但自己自由发挥的空间较少。例如代权重的最小二乘此函数就无法实现。
*
*/
//————————————————
//版权声明：本文为CSDN博主「xuelangwin」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/xuelangwin/article/details/81095651
template<typename T, int _Options>
bool LevenbergMarquardt<T,_Options>::update(const LevenbergMarquardt::VectorXT* &paramPtr, LevenbergMarquardt::MatT* &JPtr,LevenbergMarquardt::MatT* &errPtr)
{
    change=false;
    JPtr = errPtr = 0;
    assert( err.size()!=0 );
    if( state == DONE )
    {
        paramPtr =&param;
        return false;
    }
    if( state == STARTED )
    {
        paramPtr = &param;
        J.fill(0);
        err.fill(0);
        JPtr = &J;
        errPtr = &err;
        state = CALC_J;
        return true;
    }
    if( state == CALC_J )
    {
        JtJ=J.transpose()*J;
        JtErr=J.transpose()*err;
        prevParam=param;
        step();
        if( iters == 0 )
            prevErrNorm = err.norm();
        paramPtr = &param;
        err.fill(0);
        errPtr = &err;
        state = CHECK_ERR;
        return true;
    }
    assert( state == CHECK_ERR );
    errNorm = err.norm();
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {

            step();
            paramPtr = &param;
            err.fill(0);
            errPtr = &err;
            state = CHECK_ERR;
            return true;
        }
    }
    lambdaLg10 = std::max(lambdaLg10-1, -16);
    if( ++iters >= criteria.max_iter ||
            (param-prevParam).norm() < criteria.epsilon )
    {
        const double LOG10 = log(10.);
        double lambda = exp(lambdaLg10*LOG10);
        std::cout<<"lambda:"<<lambda<<std::endl;
        std::cout<<"JtJW:"<<JtJW<<std::endl;
        paramPtr = &param;
        state = DONE;
        return true;
    }
    prevErrNorm = errNorm;
    paramPtr = &param;
    J.fill(0);
    JPtr = &J;
    errPtr = &err;
    state = CALC_J;
    return true;
}


/*
    * 功能：
    * 参数：
    *   [out]      param         需要优化的参数，根据内部参数的优化状况，对当前参数进行输出，方便外部误差的计算
    *   [in]       JtJ           正规方程左侧的部分，此变量与类内部成员变量关联，在此函数后面对其赋值。可以变成J'ΩJ
    *   [in]       JtErr         正规方程右侧的部分，此变量与类内部成员变量关联，在此函数后面对其赋值。
    *   [in]       errNorm       测量误差（测量值减去计算出的测量值的模），此变量与类内部成员变量关联，在此函数后面对其赋值。
    *
    *  返回值：
    *
    *  备注：
    *      目标函数的导数，正规方程的左侧和右侧计算都是在外部完成计算的，测量误差的计算也是在外部完成的。
    */
//————————————————
//版权声明：本文为CSDN博主「xuelangwin」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/xuelangwin/article/details/81095651
template<typename T, int _Options>
bool LevenbergMarquardt<T,_Options>::updateAlt(const LevenbergMarquardt::VectorXT* &paramPtr, LevenbergMarquardt::MatT* &JPtr, LevenbergMarquardt::MatT* &JtErrPtr, T &outErrNorm)
{
    assert( err.size()!=0 );
    if( state == DONE )
    {
        paramPtr =& param;
        return false;
    }

    if( state == STARTED )
    {
        paramPtr =& param;
        JtJ.fill(0);
        JtErr.fill(0);
        errNorm = 0;
        JPtr = JtJ;
        JtErrPtr = JtErr;
        outErrNorm = errNorm;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        param=prevParam;
        step();
        paramPtr =& param;
        prevErrNorm = errNorm;
        errNorm = 0;
        outErrNorm = errNorm;
        state = CHECK_ERR;
        return true;
    }

    assert( state == CHECK_ERR );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 1 )
        {
            step();
            paramPtr =& param;
            errNorm = 0;
            outErrNorm = errNorm;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.max_iter ||
            (param-prevParam).norm() < criteria.epsilon )
    {
        paramPtr = param;
        JPtr = &JtJ;
        JtErrPtr = &JtErr;
        state = DONE;
        return false;
    }

    prevErrNorm = errNorm;
    JtJ.fill(0);
    JtErr.fill(0);
    paramPtr = &param;
    JPtr = &JtJ;
    JtErrPtr = &JtErr;
    state = CALC_J;
    return true;
}

template<typename T, int _Options>
void LevenbergMarquardt<T,_Options>::step()
{
    using namespace std;
    const double LOG10 = log(10.);
    double lambda = exp(lambdaLg10*LOG10);
    int nparams = param.rows();

    int nparams_nz = std::count_if (mask.begin(), mask.end(), [&](const unsigned char& c){
        return (c==1);
    });
    if(JtJN.rows() != nparams_nz) {
        // prevent re-allocation in every step
        JtJN.setZero(nparams_nz, nparams_nz);
        JtJV.setZero(nparams_nz, 1);
        JtJW.setZero(nparams_nz, 1);
    }
    subMatrix(JtErr, JtJV, std::vector<unsigned char>(1, 1), mask);
    subMatrix(JtJ, JtJN, mask, mask);
    if(err.size()!=0)
        completeSymm( JtJN, completeSymmFlag );
    JtJN.diagonal() *= 1. + lambda;
    JtJV.array()*=-1;
    solve(JtJN, JtJV, JtJW, solveMethod);
    int j = 0;
    for( int i = 0; i < nparams; i++ )
        param(i) = prevParam(i) + (mask[i] ? JtJW(j++) : 0);
    change=true;
}

template<typename T, int _Options>
void LevenbergMarquardt<T,_Options>::completeSymm(LevenbergMarquardt::MatT &m, bool LtoR)
{

}

template<typename T, int _Options>
bool LevenbergMarquardt<T,_Options>::solve(LevenbergMarquardt::MatT &JtJN, LevenbergMarquardt::MatT &JtJV, LevenbergMarquardt::VectorXT &JtJW, int &solveMethod)
{
    bool result;
    bool is_normal = (solveMethod & DECOMP_NORMAL) != 0;
    assert((solveMethod != DENSE_LU&&solveMethod != SPARSE_LU && solveMethod != DENSE_CHOLESKY&& solveMethod != SPARSE_CHOLESKY) ||
           is_normal || JtJN.rows() == JtJN.cols() );
    if((solveMethod == DENSE_LU ||solveMethod == SPARSE_LU || solveMethod == DENSE_CHOLESKY||solveMethod == SPARSE_CHOLESKY) && !is_normal &&
            JtJN.rows() <= 3 && JtJN.rows() == JtJN.cols() && JtJV.cols() == 1)
    {
        if(JtJN.rows()==2){
            T d=det2(JtJN);
            if( d != 0. )
            {
                T t;
                d = 1./d;
                t = ((JtJV(0)*JtJN(1,1) - JtJV(1)*JtJN(0,1))*d);
                JtJW(1) = ((JtJV(1)*JtJN(0,0) - JtJV(0)*JtJN(1,0))*d);
                JtJW(0) = t;
            }
            else
                result = false;
        }else if(JtJN.rows()==3){
            T d=det3(JtJN);
            if( d != 0. )
            {
                T t[3];
                d = 1./d;

                t[0] = (d*
                        (JtJV(0)*(JtJN(1,1)*JtJN(2,2) - JtJN(1,2)*JtJN(2,1)) -
                         JtJN(0,1)*(JtJV(1)*JtJN(2,2) - JtJN(1,2)*JtJV(2)) +
                         JtJN(0,2)*(JtJV(1)*JtJN(2,1) - JtJN(1,1)*JtJV(2))));

                t[1] = (d*
                        (JtJN(0,0)*(JtJV(1)*JtJN(2,2) - JtJN(1,2)*JtJV(2)) -
                         JtJV(0)*(JtJN(1,0)*JtJN(2,2) - JtJN(1,2)*JtJN(2,0)) +
                         JtJN(0,2)*(JtJN(1,0)*JtJV(2) - JtJV(1)*JtJN(2,0))));

                t[2] = (d*
                        (JtJN(0,0)*(JtJN(1,1)*JtJV(2) - JtJV(1)*JtJN(2,1)) -
                         JtJN(0,1)*(JtJN(1,0)*JtJV(2) - JtJV(1)*JtJN(2,0)) +
                         JtJV(0)*(JtJN(1,0)*JtJN(2,1) - JtJN(1,1)*JtJN(2,0))));

                JtJW(0) = t[0];
                JtJW(1) = t[1];
                JtJW(2) = t[2];
            }
            else
                result = false;
        }else{
            assert( JtJN.rows() == 1 );
            T d = JtJN(0,0);
            if( d != 0. )
                JtJW(0) = (JtJV(0)/d);
            else
                result = false;
        }
        return result;
    }
    switch (solveMethod) {
    case DENSE_LU:
    {
        JtJW=JtJN.fullPivLu().solve(JtJV);
        if((JtJN*JtJW).isApprox(JtJV))
        {
            std::cout << "Here is a solution x to the equation mx=y:" << std::endl << JtJW << std::endl;
            return false;
        }
        else{
            std::cout << "The equation mx=y does not have any solution." << std::endl;
            return true;
        }
    }
        break;
    case SPARSE_LU:
    {
        Eigen::SparseMatrix<T> A = JtJN.sparseView();
        Eigen::SparseMatrix<T> b = JtJV.sparseView();
        Eigen::SparseLU<Eigen::SparseMatrix<T>,Eigen::COLAMDOrdering<int>> LU;
        LU.compute(A);
        if(LU.info()!=Eigen::Success) {
            // decomposition failed
            return false;
        }
        JtJW = LU.solve(b);
        if(LU.info()!=Eigen::Success) {
            // solving failed
            return false;
        }
        return true;
    }
        break;
    case DECOMP_SVD:
    {
        Eigen::JacobiSVD<MatT> svd(JtJN, Eigen::ComputeThinU | Eigen::ComputeThinV);
        JtJW=svd.solve(JtJV);
        return true;
    }
        break;
    case DENSE_CHOLESKY:
    {
        Eigen::LDLT<MatT> LDLT;
        LDLT.compute(JtJN);
        if(LDLT.info()!=Eigen::Success) {
            // decomposition failed
            return false;
        }
        JtJW=LDLT.solve(JtJV);
        if(LDLT.info()!=Eigen::Success) {
            // solving failed
            return false;
        }
        if((JtJN*JtJW).isApprox(JtJV))
        {
            std::cout << "Here is a solution x to the equation mx=y:" << std::endl << JtJW << std::endl;
            return false;
        }
        else{
            std::cout << "The equation mx=y does not have any solution." << std::endl;
            return true;
        }
    }
        break;
    case SPARSE_CHOLESKY:
    {
        Eigen::SparseMatrix<T> A = JtJN.sparseView();
        Eigen::SparseMatrix<T> b = JtJV.sparseView();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> LDLT;
        LDLT.compute(A);
        if(LDLT.info()!=Eigen::Success) {
            // decomposition failed
            return false;
        }
        JtJW = LDLT.solve(b);
        if(LDLT.info()!=Eigen::Success) {
            // solving failed
            return false;
        }
        return true;
    }
        break;
    case DENSE_QR:
    {
        JtJW = JtJN.colPivHouseholderQr().solve(JtJV);
        return true;
    }
        break;
    case SPARSE_QR:
    {
        Eigen::SparseMatrix<T> A = JtJN.sparseView();
        Eigen::SparseMatrix<T> b = JtJV.sparseView();
        Eigen::SparseQR<Eigen::SparseMatrix<T>,Eigen::COLAMDOrdering<int>> QR;
        QR.compute(A);
        if(QR.info()!=Eigen::Success) {
            // decomposition failed
            return false;
        }
        JtJW = QR.solve(b);
        if(QR.info()!=Eigen::Success) {
            // solving failed
            return false;
        }
        return true;
    }
        break;
    case DECOMP_NORMAL:
    {
        Eigen::SparseMatrix<T> A = JtJN.sparseView();
        Eigen::SparseMatrix<T> b = JtJV.sparseView();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> LDLT;
        LDLT.compute(A.transpose()*A);
        if(LDLT.info()!=Eigen::Success) {
            // decomposition failed
            return false;
        }
        JtJW = LDLT.solve(A.transpose()*b);
        if(LDLT.info()!=Eigen::Success) {
            // solving failed
            return false;
        }
        return true;
    }
        break;
    }
}
template<typename T, int _Options>
void LevenbergMarquardt<T,_Options>::subMatrix(const MatT& src, MatT& dst, const std::vector<unsigned char>& cols,
                                               const std::vector<unsigned char>& rows) {
    //int nonzeros_cols = cv::countNonZero(cols);
    int nonzeros_cols = std::count_if (cols.begin(), cols.end(), [&](const unsigned char& c){
        return (c==1);
    });
    MatT tmp=MatT::Zero(src.rows(), nonzeros_cols);

    for (int i = 0, j = 0; i < (int)cols.size(); i++)
    {
        if (cols[i])
        {
            //src.col(i).copyTo(tmp.col(j++));
            tmp.col(j++)=src.col(i);
        }
    }

    int nonzeros_rows  = std::count_if (rows.begin(), rows.end(), [&](const unsigned char& c){
        return (c==1);
    });
    dst=MatT::Zero(nonzeros_rows, nonzeros_cols);
    for (int i = 0, j = 0; i < (int)rows.size(); i++)
    {
        if (rows[i])
        {
            //tmp.row(i).copyTo(dst.row(j++));
            dst.row(j++)=tmp.row(i);
        }
    }
}
};
#endif // LEVENBERGMARQUARDT_H
