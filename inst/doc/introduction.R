## ----eval=FALSE---------------------------------------------------------------
#  SVT = function(M,step_size,tolerance,lamda,k_max){
#    p = dim(M)[1]
#    q = dim(M)[2]
#    p_q = min(p,q)
#  
#    X = matrix(0,p,q*k_max)
#    Y = matrix(0,p,q*k_max)
#  
#    Y[1:p,1:q] = M
#  
#    for (i in 2:k_max){
#      Y_n_1 = Y[1:p,(i-1):(i+q-2)]
#      U = svd(Y_n_1)$u
#      V = svd(Y_n_1)$v
#      d = d_lamda = numeric(p_q)
#  
#      for (j in 1:p_q){
#        d[j] = svd(Y_n_1)$d[j]
#        d_lamda[j] = max((d[j] - lamda),0)
#      }
#  
#      s_d = diag(d_lamda)
#      X_n =  U%*%s_d%*%t(V)
#      X[1:p,(i+q-1):(i+2*q-2)] = X_n
#  
#      Y_n = Y_n_1 +( M - X_n)*step_size
#      Y[1:p,(i+q-1):(i+2*q-2)] = Y_n
#  
#      if( norm((X_n-M),"F")/norm(M,"F") < tolerance){
#        iters = i
#        break
#      }else{
#        iters = k_max    }
#    }
#  
#    relative_error = norm((X_n-M),"F")
#  
#    return(list(X_n,relative_error))
#  }

## ----eval=FALSE---------------------------------------------------------------
#  SVP = function(M,step_size,tolerance,rank_r,k_max){
#    p = dim(M)[1]
#    q = dim(M)[2]
#    p_q = min(p,q)
#  
#    Y = X = matrix(0,p,q*k_max)
#  
#    Y[1:p,1:q] = M
#  
#    for (i in 2:k_max){
#      Y_n_1 = Y[1:p,(i-1):(i+q-2)]
#      U = svd(Y_n_1)$u
#      V = svd(Y_n_1)$v
#      d  = svd(Y_n_1)$d
#  
#      d_r = d[1:rank_r]
#  
#      U_r = U[1:p,1:rank_r]
#      V_r = V[1:q,1:rank_r]
#      s_d = diag(d_r)
#  
#      X_n =  U_r%*%s_d%*%t(V_r)
#      X[1:p,(i+q-1):(i+2*q-2)] = X_n
#  
#      Y_n = Y_n_1 + (M - X_n)*step_size
#      Y[1:p,(i+q-1):(i+2*q-2)] = Y_n
#  
#      if( norm((X_n-M),"F")/norm(M,"F") < tolerance){
#        iters = i
#        break
#      }else{
#        iters = k_max    }
#    }
#  
#    relative_error = norm((X_n-M),"F")
#  
#    return(list(X_n,relative_error))
#  }

## -----------------------------------------------------------------------------
WSVT = function(M,w,lamda){
  p = dim(M)[1]
  q = dim(M)[2]
  p_q = min(p,q)
  
  L = d = d_lamda = numeric(p_q)
  
  for (i in 1:(p_q-1)){
    if (w[i] >= w[i+1] ){L[i] = 1
  }else{
    L[i] = 0
  }}
  
  for (j in 1:p_q){
    if (sum(L) == p_q-1){
      d[j] = svd(M)$d[j]
      d_lamda[j] = max((d[j] - lamda*w[j]),0)
    }
  }
  
  s_d = diag(d_lamda)
  U = svd(M)$u
  V = svd(M)$v
  X =  U%*%s_d%*%t(V)

  relative_error = norm((X-M),"F")

  return(list(X,relative_error))
}



## ----eval=TRUE----------------------------------------------------------------
library(StatComp21076)
set.seed(1)
M1 = matrix(rnorm(25),5,5)
M2 = matrix(rnorm(25),5,5)
M = M1%*%M2
k = c(10,100,1000)

t1 = SVT(M, step_size = 1, tolerance = 0.01, lamda = 3, k_max = k[1])[2]
p1 = SVP(M, step_size = 1, tolerance = 0.01, rank_r = 3, k_max = k[1])[2]
t2 = SVT(M, step_size = 1, tolerance = 0.01, lamda = 3, k_max = k[2])[2]
p2 = SVP(M, step_size = 1, tolerance = 0.01, rank_r = 3, k_max = k[2])[2]
t3 = SVT(M, step_size = 1, tolerance = 0.01, lamda = 3, k_max = k[3])[2]
p3 = SVP(M, step_size = 1, tolerance = 0.01, rank_r = 3, k_max = k[3])[2]
WSVT(M,w = c(0.5,0.4,0.1,0.2,0.1),lamda=3)[2]

r_t = matrix(c(t1,p1,t2,p2,t3, p3),2,3)
rownames(r_t) = c("SVT","SVP")
colnames(r_t) = paste("k_max=",k)
knitr::kable(r_t)


