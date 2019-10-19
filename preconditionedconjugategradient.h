#ifndef PRECONDITIONEDCONJUGATEGRADIENT_H
#define PRECONDITIONEDCONJUGATEGRADIENT_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
template<typename Functor,typename Vector,typename T>
class PreConditionedConjugateGradient{
public:
    enum ReturnInfo
    {
        CG_CONVERGENCE,
        CG_MAX_ITERATIONS,
        CG_INVALID_INPUT
    };
    struct Options
    {
        Options (void);
        int maxIterations;
        T tolerance;
    };

    struct Status
    {
        Status (void);
        int numIterations;
        ReturnInfo info;
    };
    Status solve(const Functor& A,const Vector& b,Vector& x,const Functor *P=nullptr);
private:
    Options opts;
    Status status;
};
template<typename Functor, typename Vector, typename T>
typename PreConditionedConjugateGradient<Functor,Vector,T>::Status PreConditionedConjugateGradient<Functor,Vector,T>::solve(const Functor &A, const Vector &b,Vector& x,const Functor *P)
{
    assert(A.rows()==A.cols());
    assert(A.cols()=b.rows());
    if(A.cols()!=x.size()){
        x.resize(A.cols());
    }
    x.fill(0);
    Vector r=b;
    Vector d;
    Vector z;
    T rkdotrk;
    if(P==nullptr){
        rkdotrk=r.squaredNorm();
        d=r;
        opts.maxIterations=A.cols();
    }else{
        z=(*P)*r;
        rkdotrk=z.dot(r);
        d=z;
    }
    Status& status=this->status;
    Options& opts=this->opts;
    for(status.numIterations=0;status.numIterations<opts.maxIterations;status.numIterations++){
        /* Compute step size in search direction. */
        Vector Ad=A*d;
        T alpha=rkdotrk/d.dot(Ad);//alpha_k=z_k.dot(r_k)/d_k.dot(A*d_k) or alpha_k=g_k.dot(g_k)/d_k.dot(A*d_k)
        /* Update parameter vector. */
        x+=alpha*d;//x_k+1=x_k+alpha_k*d_k;
        /* Compute new residual and its norm. */
        r-=Ad*alpha;// r_k+1=r_k-alpha_k*A*d_k or g_k+1=g_k+alpha_k*A*d_k
        T new_rkplusdotrkplus = r.dot(r);
        /* Check tolerance condition. */
        if (new_rkplusdotrkplus < this->opts.tolerance)//||r_k+1||^2<epsilon
        {
            this->status.info = CG_CONVERGENCE;
            return this->status;
        }
        /* Precondition residual if necessary. */
        if (P != nullptr)
        {
            z = (*P)*r;//z_k+1=P*r_k+1
            new_rkplusdotrkplus = z.dot(r);
        }
        /*
      * Update search direction.
      * The next residual will be orthogonal to new Krylov space.
      */
        T beta = new_rkplusdotrkplus/rkdotrk;//z_k+1.dot(r_k+1
        if (P != nullptr)
            d = z+beta*d;//d_k+1=z_k+1 + beta_k*d_k
        else
            d = r+beta*d;//d_k+1=r_k+1 + beta_k*d_k=-g_k+1 +beta_k*d_k

        /* Update residual norm. */
        rkdotrk = new_rkplusdotrkplus;
    }
    this->status.info = CG_MAX_ITERATIONS;
     return this->status;
}
template<typename Functor, typename Vector, typename T>
PreConditionedConjugateGradient<Functor,Vector,T>::Status::Status():numIterations(0)
{

}
template<typename Functor, typename Vector, typename T>
PreConditionedConjugateGradient<Functor,Vector,T>::Options::Options():maxIterations(100), tolerance(1e-20)
{

}
#endif // PRECONDITIONEDCONJUGATEGRADIENT_H






