#' @title SVT
#' @description A Singular Value Thresholding Algorithm for Matrix Completion
#' @param M sparse matrix
#' @param step_size step size
#' @param tolerance tolerance
#' @param lamda tunning parameter
#' @param k_max maximum number of iterations
#' @return complete matrix 
#' @examples
#' \dontrun{
#' M <- matrix(c(0,1,0,1,0,1,0,1,0),3,3)
#' f <- SVT(M,1,0.9,1,10000)
#' }
#' @export
SVT = function(M,step_size,tolerance,lamda,k_max){
  p = dim(M)[1]
  q = dim(M)[2]
  p_q = min(p,q)
  X = matrix(0,p,q*k_max)
  Y = matrix(0,p,q*k_max)
  Y[1:p,1:q] = M
  for (i in 2:k_max){
    Y_n_1 = Y[1:p,(i-1):(i+q-2)]
    U = svd(Y_n_1)$u
    V = svd(Y_n_1)$v
    d = d_lamda = numeric(p_q)
    for (j in 1:p_q){
      d[j] = svd(Y_n_1)$d[j]
      d_lamda[j] = max((d[j] - lamda),0)
    }
    s_d = diag(d_lamda)
    X_n =  U%*%s_d%*%t(V)
    X[1:p,(i+q-1):(i+2*q-2)] = X_n
    Y_n = Y_n_1 +( M - X_n)*step_size
    Y[1:p,(i+q-1):(i+2*q-2)] = Y_n
    if( norm((X_n-M),"F")/norm(M,"F") < tolerance){
      iters = i
      break
    }else{
      iters = k_max }
  }
  relative_error = norm((X_n-M),"F")
  return(list(X_n,relative_error))
}

#' @title SVP
#' @description Guaranteed Rank Minimization via Singular Value Projection
#' @param M sparse matrix
#' @param step_size step size
#' @param tolerance tolerance
#' @param rank_r Rank Minimization
#' @param k_max maximum number of iterations
#' @return complete matrix 
#' @examples
#' \dontrun{
#' M <- matrix(c(0,1,0,1,0,1,0,1,0),3,3)
#' g <- SVP(M,1,0.01,2,10000)
#' }
#' @export
SVP = function(M,step_size,tolerance,rank_r,k_max){
  p = dim(M)[1]
  q = dim(M)[2]
  p_q = min(p,q)
  Y = X = matrix(0,p,q*k_max)
  Y[1:p,1:q] = M
  for (i in 2:k_max){
    Y_n_1 = Y[1:p,(i-1):(i+q-2)]
    U = svd(Y_n_1)$u
    V = svd(Y_n_1)$v
    d  = svd(Y_n_1)$d
    d_r = d[1:rank_r]
    U_r = U[1:p,1:rank_r]
    V_r = V[1:q,1:rank_r]
    s_d = diag(d_r)
    X_n =  U_r%*%s_d%*%t(V_r)
    X[1:p,(i+q-1):(i+2*q-2)] = X_n
    Y_n = Y_n_1 + (M - X_n)*step_size
    Y[1:p,(i+q-1):(i+2*q-2)] = Y_n
    if( norm((X_n-M),"F")/norm(M,"F") < tolerance){
      iters = i
      break
    }else{
      iters = k_max    }
  }
  relative_error = norm((X_n-M),"F")
  return(list(X_n,relative_error))
}

#' @title WSVT
#' @description Nonconvex Nonsmooth Low Rank Minimization via Iteratively Reweighted Nuclear Norm
#' @param M sparse matrix
#' @param lamda tunning parameter
#' @param w weight 
#' @return complete matrix 
#' @examples
#' \dontrun{
#' M <- matrix(c(0,1,0,1,0,1,0,1,0),3,3)
#' g <- SVP(M,1,0.01,2,10000)
#' }
#' @export
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
  adaptive =  U%*%s_d%*%t(V)
  
  relative_error = norm((adaptive-M),"F")
  
  return(list(adaptive,relative_error))
}
