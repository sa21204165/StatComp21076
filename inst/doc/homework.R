## ----eval=FALSE---------------------------------------------------------------
#  ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
#  trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
#  group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
#  weight <- c(ctl, trt)
#  lm.D9 <- lm(weight ~ group)
#  par(mfrow=c(2,2))
#  plot(lm.D9)

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  
#  X = array(rnorm(25),dim = c(5,5))
#  X

## ----eval=FALSE---------------------------------------------------------------
#  par(mfrow = c(2,2))
#  n = 10
#  sigma = 1
#  U = runif(n, 0, 1)
#  X = sqrt(-2*log(1-U))*sigma
#  hist(X, main = "Histogram of Rayleigh (sigma = 1)")
#  abline(v = sigma, col = "red")
#  sigma = 2
#  U = runif(n, 0, 1)
#  X = sqrt(-2*log(1-U))*sigma
#  hist(X, main = "Histogram of Rayleigh (sigma = 2)")
#  abline(v = sigma, col = "red")
#  sigma = 0.5
#  U = runif(n, 0, 1)
#  X = sqrt(-2*log(1-U))*sigma
#  hist(X, main = "Histogram of Rayleigh (sigma = 0.5)")
#  abline(v = sigma, col = "red")
#  sigma = 5
#  U = runif(n, 0, 1)
#  X = sqrt(-2*log(1-U))*sigma
#  hist(X, main = "Histogram of Rayleigh (sigma = 5)")
#  abline(v = sigma, col = "red")

## ----eval=FALSE---------------------------------------------------------------
#  n = 1000
#  samp = function(p1){
#    U = runif(1, 0, 1)
#    if(U < p1){
#      x = rnorm(1,0,1)
#    }else{
#      x = rnorm(1,3,1)
#    }
#    return(x)
#  }
#  pdf = function(x, p1 = p1){
#    p1*dnorm(x, 0, 1) + (1-p1)*dnorm(x, 3, 1)
#  }
#  
#  par(mfrow = c(2,2))
#  p1 = 0.75
#  X = replicate(n, samp(p1 = p1))
#  hist(X,breaks = 50, xlim = c(-4, 6), prob = T, main = "Histogram of  Mix-Gaussian (p = 0.75)")
#  curve(pdf(x, p1 = p1), -4, 6, col = "red", add = T)
#  
#  p1 = 0.5
#  X = replicate(n, samp(p1 = p1))
#  hist(X,breaks = 50, xlim = c(-4, 6), prob = T, main = "Histogram of  Mix-Gaussian (p = 0.5)")
#  curve(pdf(x, p1 = p1), -4, 6, col = "red", add = T)
#  
#  p1 = 0.25
#  X = replicate(n, samp(p1 = p1))
#  hist(X,breaks = 50, xlim = c(-4, 6), prob = T, main = "Histogram of  Mix-Gaussian (p = 0.25)")
#  curve(pdf(x, p1 = p1), -4, 6, col = "red", add = T)
#  
#  p1 = 0.9
#  X = replicate(n, samp(p1 = p1))
#  hist(X,breaks = 50, xlim = c(-4, 6), prob = T, main = "Histogram of  Mix-Gaussian (p = 0.9)")
#  curve(pdf(x, p1 = p1), -4, 6, col = "red", add = T)

## ----eval=FALSE---------------------------------------------------------------
#  poi_gamm_simu <- function(lambda, alpha, beta, t){
#    f = function(lambda = lambda, t = t, alpha = alpha, beta = beta){
#      N = rpois(1, lambda*t)
#      X = sum(rgamma(N, alpha, beta))
#    }
#    Xt = replicate(10, f(lambda = lambda, t = t, alpha = alpha, beta = beta))
#    result = matrix(0,2,2)
#    result[1,1] = mean(Xt)
#    result[2,1] = var(Xt)
#    result[1,2] = lambda*t*alpha/beta
#    result[2,2] = lambda*t*(alpha+alpha^2)/beta^2
#    rownames(result) = c("E[X(t)]", "Var(X(t))")
#    colnames(result) = c("Estimate", "Theoretical")
#    result
#  
#  }
#  poi_gamm_simu(lambda = 1, alpha = 2, beta = 4, t = 10)
#  poi_gamm_simu(lambda = 1, alpha = 4, beta = 2, t = 10)
#  poi_gamm_simu(lambda = 2, alpha = 3, beta = 5, t = 10)
#  poi_gamm_simu(lambda = 2, alpha = 5, beta = 3, t = 10)

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(21076)
#  m = 10
#  x = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
#  y = runif(m,min = 0,max = x)
#  t = mean(y^2*(1-y)^2)*30*x
#  z = pbeta(x,3,3)
#  print(c(t))
#  print(c(z))

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(1)
#  sigma = 1
#  u = runif(1,0,1)
#  x = sqrt(-2*log(1-u))*sigma
#  m = 10
#  y = runif(m, 0, x)
#  VAR_x = var(c(x*y/sigma*exp(-y^2/2/(sigma)^2)))/m
#  t = runif(m/2, 0, x)
#  VAR_t = var(c(x*t/sigma*exp(-t^2/2/(sigma)^2)+x*(x-t)/sigma*exp(-(x-t)^2/2/(sigma)^2)))/m
#  print(VAR_x)
#  print(VAR_t)

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(2)
#  n = 10
#  y = runif(n,0,1/2)
#  var_t = var(1/2*sqrt(y)/sqrt(2*pi)*exp(-y))
#  u = runif(n,0,1)
#  x = u^{1/3}
#  var_f = var(1/3/sqrt(2*pi)*exp(-x^2/2))
#  print(c(var_t))
#  print(c(var_f))

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(7)
#  n = 20
#  m = 10
#  alpha = 0.05
#  x = rchisq(n,2)
#  theta_1=replicate(m,expr = {
#    x = rchisq(n,2)
#    mean(x)-sd(x)/sqrt(n)*qt((1-alpha/2),n-1)
#    })
#  theta_2=replicate(m,expr = {
#    x = rchisq(n,2)
#    mean(x)+sd(x)/sqrt(n)*qt((1-alpha/2),n-1)
#    })
#  
#  d = sum((theta_1 < 2) & (2 < theta_2))
#  print(c(d/m))

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(7)
#  n = 20
#  m = 100
#  alpha = 0.05
#  mu_0 = 1
#  p_1 = p_2 = p_3 =numeric(m)
#  x_1 = rchisq(n,1)
#  x_2 = runif(n,0,2)
#  x_3 = rexp(n,1)
#  for(i in 1:m){
#    x_1 = rchisq(n,1)
#    ttest_1 = t.test(x_1,alternative = "greater",mu = mu_0)
#    p_1[i] = ttest_1$p.value
#  }
#  for(i in 1:m){
#    x_2 = runif(n,0,2)
#    ttest_2 = t.test(x_2,alternative = "greater",mu = mu_0)
#    p_2[i] = ttest_2$p.value
#  }
#  for(i in 1:m){
#    x_3 = rexp(n,1)
#    ttest_3 = t.test(x_3,alternative = "greater",mu = mu_0)
#    p_3[i] = ttest_3$p.value
#  }
#  p_1.hat = mean(p_1 < alpha)
#  p_2.hat = mean(p_2 < alpha)
#  p_3.hat = mean(p_3 < alpha)
#  print(c(p_1.hat,p_2.hat,p_3.hat))

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(12)
#  library(MASS)
#  mean = c(0,0)
#  sigma = matrix(c(1,0,0,1),nrow = 2,ncol = 2)
#  
#  
#  n = c(1,2,3,5,10,50,60)
#  d = c(2,2,2,2,2,2,2)
#  cv = qchisq(0.975,0,d*(d+1)*(d+2)/6)*6/n
#  print(cv)
#  
#  sk = function(x){
#    xbar = mean(x)
#    m3 = mean((x-xbar)^3)
#    m2 = mean((x-xbar)^2)
#    return(m3/m2^1.5)
#    }
#  
#  p.reject = numeric(length(n))
#  m = 100
#  for (i in 1:length(n)){
#    sktests = numeric(m)
#    for (j in 1:m){
#      x = mvrnorm(n[i],mean,sigma)
#      sktests[j] = as.integer(abs(sk(x)) >= cv[i] )
#    }
#    p.reject[i] = mean(sktests)
#    }
#  
#  print(p.reject)
#  

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(13)
#  library(MASS)
#  mean1 = c(0,0)
#  sigma1 = matrix(c(1,0,0,1),nrow = 2,ncol = 2)
#  mean2 = c(0,0)
#  sigma2 = matrix(c(100,0,0,100),nrow = 2,ncol = 2)
#  
#  alpha = 0.1
#  n = 3
#  m = 25
#  epsilon = c(seq(0,0.15,0.01),seq(0.15,1,0.05))
#  N = length(epsilon)
#  pwr = numeric(N)
#  d = 2
#  cv = qchisq(1-alpha/2,d*(d+1)*(d+2)/6)*6/n
#  
#  sk = function(x){
#    xbar = mean(x)
#    m3 = mean((x-xbar)^3)
#    m2 = mean((x-xbar)^2)
#    return(m3/m2^1.5)
#  }
#  
#  for(j in 1:N){
#    e = epsilon[j]
#    sktests = numeric(m)
#    for(i in 1:m){
#      index = sample(c(1,10),replace = TRUE,size = n,prob = c(1-e,e))
#      x = matrix(0,nrow = n,ncol = 2)
#      for (t in 1:n){
#        if(index[t] == 1) x[t,] = mvrnorm(1,mean1,sigma1)
#        else x[t,] = mvrnorm(1,mean2,sigma2)
#      }
#      sktests[i] = as.integer(abs(sk(x)) >= cv)
#    }
#    pwr[j] = mean(sktests)
#  }
#  
#  plot(epsilon,pwr,type = "b",xlab = bquote(epsilon),ylim = c(0,1))
#  abline(h = 0.1,lty = 3)
#  se = sqrt(pwr*(1-pwr)/m)
#  lines(epsilon,pwr+se,lty = 3)
#  lines(epsilon,pwr-se,lty = 3)

## ----eval=FALSE---------------------------------------------------------------
#  library(bootstrap)
#  set.seed(321)
#  
#  B = 100
#  n = nrow(scor)
#  theta_hat_star = numeric(B)
#  
#  lambda_hat = eigen(cov(scor))$values
#  theta_hat = lambda_hat[1] / sum(lambda_hat)
#  
#  for (b in 1:B){
#    scor_star = sample(scor,replace = TRUE)
#    lambda_hat_star = eigen(cov(scor_star))$values
#    theta_hat_star[b] = lambda_hat_star[1] / sum(lambda_hat_star)
#  }
#  
#  se_theta_hat_star = sd(theta_hat_star)
#  bias_theta_hat_star =  mean(theta_hat_star) - theta_hat
#  
#  print(bias_theta_hat_star)
#  print(se_theta_hat_star)
#  hist(theta_hat_star,prob = TRUE)
#  
#  
#  
#  

## ----eval=FALSE---------------------------------------------------------------
#  
#  library(bootstrap)
#  set.seed(321)
#  
#  n = nrow(scor)
#  lambda_hat = eigen(cov(scor))$values
#  theta_hat = lambda_hat[1] / sum(lambda_hat)
#  theta_hat_star = numeric(n)
#  
#  for (i in 1:n) {
#  scor_star = scor [-i,]
#  lambda_hat_star = eigen(cov(scor_star))$values
#  theta_hat_star[i] = lambda_hat_star[1] / sum(lambda_hat_star)
#  }
#  
#  bias_jack = (n-1)*(mean(theta_hat_star)-theta_hat)
#  se_jack = (n-1)*sqrt(var(theta_hat_star)/n)
#  
#  print(bias_jack)
#  print(se_jack)

## ----eval=FALSE---------------------------------------------------------------
#  library(boot)
#  set.seed(321)
#  data(scor,package = "bootstrap")
#  theta.boot = function(dat,i){
#    scor_star = sample(dat,replace = TRUE)
#    lambda_hat_star = eigen(cov(scor_star))$values
#    theta_hat_star[i] = lambda_hat_star[1] / sum(lambda_hat_star)
#  }
#  
#  
#  boot.obj = boot(scor,statistic = theta.boot,R = 1000)
#  
#  boot.ci(boot.obj,type = c("perc","bca"))
#  

## ----eval=FALSE---------------------------------------------------------------
#  library(boot)
#  set.seed(321)
#  n = 100
#  x = rnorm(n,0,2)
#  y = rchisq(n,5)
#  
#  skewness_stat = function(x,i){
#    x = rnorm(n,0,2)
#    x_bar = mean(x)
#  }
#  
#  chisq_stat = function(y,i){
#    y = rchisq(n,5)
#    y_bar = mean(y)
#  }
#  
#  boot.obj_1 = boot(x,statistic = skewness_stat,R = 100)
#  boot.obj_2 = boot(x,statistic = chisq_stat,R = 100)
#  
#  boot.ci(boot.obj_1,type = c("basic","norm","perc"))
#  boot.ci(boot.obj_2,type = c("basic","norm","perc"))

## ----eval=FALSE---------------------------------------------------------------
#  library(boot)
#  set.seed(123)
#  n = 10
#  p = 5
#  q = 5
#  
#  x=c()#生成一个空的向量用于储存生成的随机数
#  for (i in 1:n){
#   xi = runif(p,0,1)#生成x的第i个p维样品
#   x = c(x,xi)
#  }
#  x = matrix(x,nrow = n,byrow = TRUE)#将向量x转化为矩阵x
#  
#  y=c()#生成一个空的向量用于储存生成的随机数
#  for (i in 1:n){
#   yi = rnorm(q,0,1)#生成y的第i个p维样品
#   y = c(y,yi)
#  }
#  y = matrix(y,nrow = n,byrow = TRUE)#将向量y转化为矩阵y
#  
#  cor.test(x,y,method = "spearman")

## ----eval=FALSE---------------------------------------------------------------
#  library(RANN)
#  library(boot)
#  library(Ball)
#  library(energy)
#  library(MASS)
#  
#  Tn = function(z, ix, sizes,k) {
#    n1 = sizes[1]; n2 = sizes[2]; n = n1 + n2
#    if(is.vector(z)) z = data.frame(z,0);
#    z = z[ix, ];
#    NN = nn2(data=z, k=k+1)
#    block1 = NN$nn.idx[1:n1,-1]
#    block2 = NN$nn.idx[(n1+1):n,-1]
#    i1 = sum(block1 < n1 + .5); i2 = sum(block2 > n1+.5)
#    (i1 + i2) / (k * n)
#  }
#  
#  eqdist.nn = function(z,sizes,k){
#    boot.obj = boot(data=z,statistic=Tn,R=R, sim = "permutation", sizes = sizes,k=k)
#    ts = c(boot.obj$t0,boot.obj$t)
#    p.value = mean(ts>=ts[1])
#    list(statistic=ts[1],p.value=p.value)
#  }

## ----eval=FALSE---------------------------------------------------------------
#  mu1 = c(0,0)
#  sigma1 = matrix(c(1,0,0,1),nrow=2,ncol=2)
#  mu2 = c(0,0)
#  sigma2 = matrix(c(2,0,0,2),nrow=2,ncol=2)
#  n1=n2=20
#  n = n1+n2
#  N = c(n1,n2)
#  k=3
#  R=999
#  m=100
#  set.seed(123)
#  p.values = matrix(NA,m,3)
#  for(i in 1:m){
#    mydata1 = mvrnorm(n1,mu1,sigma1)
#    mydata2 = mvrnorm(n2,mu2,sigma2)
#    mydata = rbind(mydata1,mydata2)
#    p.values[i,1] = eqdist.nn(mydata,N,k)$p.value
#    p.values[i,2] = eqdist.etest(mydata,sizes=N,R=R)$p.value
#    p.values[i,3] = bd.test(x=mydata1,y=mydata2,num.permutations=R,seed=i*2846)$p.value
#  }
#  alpha = 0.05;
#  pow = colMeans(p.values<alpha)
#  pow

## ----eval=FALSE---------------------------------------------------------------
#  mu1 = c(0,0)
#  sigma1 = matrix(c(1,0,0,1),nrow=2,ncol=2)
#  mu2 = c(1,1)
#  sigma2 = matrix(c(2,0,0,2),nrow=2,ncol=2)
#  n1=n2=20
#  n = n1+n2
#  N = c(n1,n2)
#  k=3
#  R=999
#  m=100
#  set.seed(123)
#  p.values = matrix(NA,m,3)
#  for(i in 1:m){
#    mydata1 = mvrnorm(n1,mu1,sigma1)
#    mydata2 = mvrnorm(n2,mu2,sigma2)
#    mydata = rbind(mydata1,mydata2)
#    p.values[i,1] = eqdist.nn(mydata,N,k)$p.value
#    p.values[i,2] = eqdist.etest(mydata,sizes=N,R=R)$p.value
#    p.values[i,3] = bd.test(x=mydata1,y=mydata2,num.permutations=R,seed=i*2846)$p.value
#  }
#  alpha = 0.05;
#  pow = colMeans(p.values<alpha)
#  pow

## ----eval=FALSE---------------------------------------------------------------
#  n1=n2=20
#  n = n1+n2
#  N = c(n1,n2)
#  k=3
#  R=999
#  m=100
#  set.seed(123)
#  p.values = matrix(NA,m,3)
#  for(i in 1:m){
#    mydata1 = as.matrix(rt(n1,1,2),ncol=1)
#    mydata2 = as.matrix(rt(n2,2,5),ncol=1)
#    mydata = rbind(mydata1,mydata2)
#    p.values[i,1] = eqdist.nn(mydata,N,k)$p.value
#    p.values[i,2] = eqdist.etest(mydata,sizes=N,R=R)$p.value
#    p.values[i,3] = bd.test(x=mydata1,y=mydata2,num.permutations=R,seed=i*2846)$p.value
#  }
#  alpha = 0.05;
#  pow = colMeans(p.values<alpha)
#  pow

## ----eval=FALSE---------------------------------------------------------------
#  n1=n2=20
#  n = n1+n2
#  N = c(n1,n2)
#  k=3
#  R=999
#  m=100
#  set.seed(123)
#  p.values = matrix(NA,m,3)
#  rbimodel=function(n,mu1,mu2,sd1,sd2){
#    index=sample(1:2,n,replace=TRUE)
#    x=numeric(n)
#    index1=which(index==1)
#    x[index1]=rnorm(length(index1), mu1, sd1)
#    index2=which(index==2)
#    x[index2]=rnorm(length(index2), mu2, sd2)
#    return(x)
#  }
#  for(i in 1:m){
#    mydata1 = as.matrix(rbimodel(n1,0,0,1,2),ncol=1)
#    mydata2 = as.matrix(rbimodel(n2,1,1,1,2),ncol=1)
#    mydata = rbind(mydata1,mydata2)
#    p.values[i,1] = eqdist.nn(mydata,N,k)$p.value
#    p.values[i,2] = eqdist.etest(mydata,sizes=N,R=R)$p.value
#    p.values[i,3] = bd.test(x=mydata1,y=mydata2,num.permutations=R,seed=i*2846)$p.value
#  }
#  alpha = 0.05;
#  pow = colMeans(p.values<alpha)
#  pow

## ----eval=FALSE---------------------------------------------------------------
#  mu1 = c(0,0)
#  sigma1 = matrix(c(1,0,0,1),nrow=2,ncol=2)
#  mu2 = c(1,1)
#  sigma2 = matrix(c(2,0,0,2),nrow=2,ncol=2)
#  n1=10
#  n2=100
#  n = n1+n2
#  N = c(n1,n2)
#  k=3
#  R=999
#  m=100
#  set.seed(123)
#  p.values = matrix(NA,m,3)
#  for(i in 1:m){
#    mydata1 = mvrnorm(n1,mu1,sigma1)
#    mydata2 = mvrnorm(n2,mu2,sigma2)
#    mydata = rbind(mydata1,mydata2)
#    p.values[i,1] = eqdist.nn(mydata,N,k)$p.value
#    p.values[i,2] = eqdist.etest(mydata,sizes=N,R=R)$p.value
#    p.values[i,3] = bd.test(x=mydata1,y=mydata2,num.permutations=R,seed=i*2846)$p.value
#  }
#  alpha = 0.05;
#  pow = colMeans(p.values<alpha)
#  pow

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  f = function(x,eta,theta){
#    stopifnot(theta > 0)
#    return(1/theta/pi/(1+(x-eta)^2/theta^2))
#  }
#  m = 10000
#  x = numeric(m)
#  
#  
#  eta = 0
#  theta = 1
#  
#  x[1] = rnorm(1,0,1)
#  k = 0
#  u = runif(m)
#  for (i in 2:m){
#    xt = x[i-1]
#    y = rnorm(1,xt,1)
#    num = f(y,eta,theta)*dnorm(xt, y,1)
#    den = f(xt,eta,theta)*dnorm(y,xt,1)
#    if (u[i] <= num/den) x[i] = y else{
#      x[i] = xt
#      k = k+1
#    }
#  }
#  
#  print(k)
#  
#  b = 1001
#  y = x[b:m]
#  a = ppoints(100)
#  QR =
#  Q = quantile(x,a)
#  qqplot(QR,Q,main = "",xlab = "Cauchy Quantiles",ylab = "Sample Quantules")
#  hist(y,breaks = "scott",main = "",xlab = "",freq = FALSE)
#  lines(QR,f(QR,0,1))
#  

## ----eval=FALSE---------------------------------------------------------------
#  
#  set.seed(123)
#  Gelman.Rubin = function(psi) {
#  psi = as.matrix(psi)
#  n = ncol(psi)
#  k = nrow(psi)
#  psi.means = rowMeans(psi)
#  B = n * var(psi.means)
#  psi.w = apply(psi, 1, "var")
#  W = mean(psi.w)
#  v.hat = W*(n-1)/n + (B/n)
#  r.hat = v.hat / W
#  return(r.hat)
#  }
#  normal.chain = function(sigma, N, X1) {
#  x = rep(0, N)
#  x[1] = X1
#  u = runif(N)
#  for (i in 2:N) {
#  xt = x[i-1]
#  y = rnorm(1, xt, sigma)
#  r1 = dcauchy(y, 0, 1) * dnorm(xt, y, sigma)
#  r2 = dcauchy(xt, 0, 1) * dnorm(y, xt, sigma)
#  r = r1 / r2
#  if (u[i] <= r) x[i] = y else
#  x[i] = xt
#  }
#  return(x)
#  }
#  
#  k = 4
#  n = 150
#  b = 10
#  x0 = c(-10, -5, 5, 10)
#  
#  sigma = 1.5
#  X = matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#  X[i, ] = normal.chain(sigma, n, x0[i])
#  psi = t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#  psi[i,] = psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  par(mfrow=c(1,1))
#  rhat = rep(0, n)
#  for (j in (b+1):n)
#  rhat[j] = Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="sigma = 1.5", ylab="R")
#  abline(h=1.1, lty=2)
#  
#  sigma = 0.5
#  X = matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#  X[i, ] = normal.chain(sigma, n, x0[i])
#  psi = t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#  psi[i,] = psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  par(mfrow=c(1,1))
#  rhat = rep(0, n)
#  for (j in (b+1):n)
#  rhat[j] = Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="sigma = 0.5", ylab="R")
#  abline(h=1.1, lty=2)
#  
#  sigma = 1
#  X = matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#  X[i, ] = normal.chain(sigma, n, x0[i])
#  psi = t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#  psi[i,] = psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  par(mfrow=c(1,1))
#  rhat = rep(0, n)
#  for (j in (b+1):n)
#  rhat[j] = Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="sigma = 1", ylab="R")
#  abline(h=1.1, lty=2)
#  

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  a = b = 1
#  n = 5
#  N = 50
#  burn = 10
#  X = matrix(0,N,2)
#  
#  X[1,] = c(2,0.2)
#  for (i in 2:N){
#    y = X[i-1,2]
#    X[i,1] = rbinom(1,n,y)#边缘分布
#    x = X[i,1]
#    X[i,2] = rbeta(1,x+a,n-x+b)#边缘分布
#  }
#  
#  b_bar = burn+1
#  X_bar = X[b_bar:N,]
#  
#  plot(X_bar,xlab = "x",ylab = "y")

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  Gelman.Rubin = function(psi) {
#  psi = as.matrix(psi)
#  n = ncol(psi)
#  k = nrow(psi)
#  psi.means = rowMeans(psi)
#  B = n * var(psi.means)
#  psi.w = apply(psi, 1, "var")
#  W = mean(psi.w)
#  v.hat = W*(n-1)/n + (B/n)
#  r.hat = v.hat / W
#  return(r.hat)
#  }
#  normal.chain = function(sigma, N, X1) {
#  x = rep(0, N)
#  x[1] = X1
#  u = runif(N)
#  for (i in 2:N) {
#  xt = x[i-1]
#  y = rnorm(1, xt, sigma)
#  r1 = dnorm(y, 0, 1) * dnorm(xt, y, sigma)
#  r2 = dnorm(xt, 0, 1) * dnorm(y, xt, sigma)
#  r = r1 / r2
#  if (u[i] <= r) x[i] = y else
#  x[i] = xt
#  }
#  return(x)
#  }
#  
#  k = 4
#  n = 150
#  b = 10
#  x0 = c(-10, -5, 5, 10)
#  
#  sigma = 0.2
#  X = matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#  X[i, ] = normal.chain(sigma, n, x0[i])
#  psi = t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#  psi[i,] = psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  par(mfrow=c(1,1))
#  rhat = rep(0, n)
#  for (j in (b+1):n)
#  rhat[j] = Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="sigma = 1.5", ylab="R")
#  abline(h=1.1, lty=2)
#  
#  sigma = 2
#  X = matrix(0, nrow=k, ncol=n)
#  for (i in 1:k)
#  X[i, ] = normal.chain(sigma, n, x0[i])
#  psi = t(apply(X, 1, cumsum))
#  for (i in 1:nrow(psi))
#  psi[i,] = psi[i,] / (1:ncol(psi))
#  print(Gelman.Rubin(psi))
#  par(mfrow=c(1,1))
#  rhat = rep(0, n)
#  for (j in (b+1):n)
#  rhat[j] = Gelman.Rubin(psi[,1:j])
#  plot(rhat[(b+1):n], type="l", xlab="sigma = 0.5", ylab="R")
#  abline(h=1.1, lty=2)
#  
#  

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  #(a)
#  func_k = function(a,k){
#    d = length(a)
#    b = sum((a)^(2*k+2))
#    e = prod(1:k)
#    c = b^(1/(2*k+2))#求欧式范数
#    g = exp(lgamma((d+1)/2)+lgamma(k+3/2)-lgamma(k+d/2+1))
#    h = (-1/2)^k/e*b/(2*k+1)/(2*k+2)*g
#    return(h)
#  }
#  
#  #(b)
#  
#  s = function(n){
#    l = 0
#    for (k in 1:n){
#      l = l + func_k(a,k)
#    }
#    return(l)#取前n项和进行进似
#  }
#  
#  #(c)
#  
#  n = 100
#  a = c(1,2)
#  print(s(n))

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  l = function(u,k0){
#    (1+(u^2)/(k0-1))^(-k0/2)
#  }
#  g = function(a,k){
#    up = sqrt(a^2 * (k-1)/(k-a^2))
#    w = integrate(l,lower=0,upper=up,k0 = k)$value
#    (2*gamma(k/2)/(sqrt(pi*(k-1))*gamma((k-1)/2)))*w
#  }
#  f = function(a,k){
#   g(a,k+1)-g(a,k)
#  }
#  f1 = function(a,k){
#    C = numeric(length(a))
#    for(i in 1:length(a)){
#    C[i] = f(a[i],k)
#    }
#    return(C)
#  }
#  k = c(16:25,50,100)
#  sol = function(k1){
#   m =uniroot(f,k=k1,lower = 1,upper = 2)$root
#   m
#  }
#  B = numeric(length(k))
#  for (i in 1:length(k)){
#  B[i] = sol(k[i])
#  }
#  
#  S = function(a,k){
#   ck = sqrt(a^2*k/(k+1-a^2))
#   pt(ck,df=k,lower.tail=FALSE)
#  }
#  
#  solve = function(k){
#    output = uniroot(function(a){S(a,k)-S(a,k-1)},lower=1,upper=2)
#    output$root
#  }
#  
#  root = matrix(0,3,length(k))
#  
#  for (i in 1:length(k)){
#    root[2,i]=round(solve(k[i]),4)
#  }
#  
#  root[1,] = k
#  root[3,] = B
#  rownames(root) = c('k','A(k)','f(k)')
#  root
#  
#  
#  

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  lambda = c(0.7, 0.3)
#  lam = sample(lambda, size = 2000, replace = TRUE)
#  y = rexp(2, rate = 1/(2*lam))
#  N = 10
#  L = c(0.7, 0.3)
#  tol = .Machine$double.eps^0.5
#  L.old =L+1
#  for (j in 1:N) {
#    f1 = dexp(y,  rate=1/(2*L[1]))
#    f2 = dexp(y,  rate=1/(2*L[2]))
#    py = f1 / (f1 + f2 )
#    qy = f2 / (f1 + f2 )
#    mu1 = sum(y * py) / sum(py)
#    mu2 = sum(y * qy) / sum(qy)
#    L = c(mu1, mu2)
#    L = L / sum(L)
#    if (sum(abs(L - L.old)/L.old) < tol) break
#    L.old = L
#  }
#  
#  print(list(lambda = L/sum(L), iter = j, tol = tol))

## ----eval=FALSE---------------------------------------------------------------
#  trims = c(0, 0.1, 0.2, 0.5)
#  x = rcauchy(100)
#  lapply(trims, function(trim) mean(x, trim = trim))##trim表示去掉异常值的比例，从trims直接一步映射到function求解去掉异常值比例后的100个柯西分布数据的均值
#  lapply(trims, mean, x = x)##先从trims映射到mean函数，均值为c（0，0.1，0.2，0.5），再将100柯西分布生成的随机数带入
#  ##两种方法只是计算的执行顺序不同，且每次迭代都是与其他所有迭代隔离的，所以计算顺序并不重要，结果一样

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  attach(mtcars)
#  
#  ##练习3
#  formulas = list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )
#  
#  out1 = vector("list", length(formulas))##for循环
#  for (i in seq_along(formulas)){
#    out1[[i]] = lm(formulas[[i]], data = mtcars)
#  }
#  out1
#  
#  l1 = lapply(formulas, function(x) lm(formula = x, data = mtcars))##lapply()
#  l1
#  
#  rsq = function(mod) summary(mod)$r.squared
#  r_square1 = lapply(l1, rsq)
#  r_square1
#  r_square2 = lapply(out1, rsq)
#  r_square2
#  
#  ##练习4
#  bootstraps = lapply(1:10, function(i) {
#  rows = sample(1:nrow(mtcars), rep = TRUE)
#  mtcars[rows, ]
#  })
#  
#  out2 = vector("list", length(formulas))##for循环
#  for (i in seq_along(formulas)){
#    out2[[i]] = lm(formulas[[i]], data = bootstraps)
#  }
#  out2[1]
#  
#  l2 = lapply(formulas, function(x) lm(formula = x, data = bootstraps))##lapply()
#  l2[1]
#  
#  rsq = function(mod) summary(mod)$r.squared
#  r_square1 = lapply(l2[1], rsq)
#  r_square1
#  r_square2 = lapply(out2[1], rsq)
#  r_square2

## ----eval=FALSE---------------------------------------------------------------
#  set.seed(123)
#  
#  ##a
#  data1 = list(x=runif(10),y=rnorm(10),z = rcauchy(10))
#  sd1 = vapply(data1,sd,FUN.VALUE = c("st"=0))
#  sd1
#  
#  ##b
#  data2 = list(x=rexp(10),y=rt(10,df = 1),z = letters[1:10])
#  
#  judge = vapply(data2, is.numeric,logical(1))
#  judge
#  
#  data3 = list()
#  for (i in 1:length(judge)){
#    if (judge[i] == TRUE){data3[i]=data2[i]}
#    else{delete.response(data3[i])}
#    }
#  
#  sd2 = vapply(data3,sd,FUN.VALUE = c("st"=0))
#  sd2
#  

## ----eval=FALSE---------------------------------------------------------------
#  library(parallel)
#  set.seed(123)
#  formulas = list(l1 = function(mtcars) glm(mtcars$mpg~mtcars$disp),l2=function(mtcars) glm(mtcars$mpg~I(1/mtcars$disp)),l3=function(mtcars) glm(mtcars$mpg~mtcars$disp+mtcars$wt),l4=function(mtcars) glm(mtcars$mpg~I(1/mtcars$disp)+mtcars$wt))
#  cl = makeCluster(4)
#  bootstraps1 = lapply(1:1000, function(i){
#    rows = sample(1:nrow(mtcars),rep = TRUE)
#    mtcars[rows,]
#  })
#  system.time(sapply(bootstraps1, formulas[[1]]))
#  system.time(parSapply(cl,bootstraps1, formulas[[1]]))

## ----eval=FALSE---------------------------------------------------------------
#  library(Rcpp)
#  library(microbenchmark)
#  set.seed(123)
#  #Gibbs2 = function(N, X0){
#    #a = 1
#    #b = 1
#    #X = matrix(0, N, 2)
#    #X[1,] = X0
#    #for(i in 2:N){
#      #X2 =  X[i-1, 2]
#      #X[i,1] = rbinom(1,25,X2)
#      #X1 = X[i,1]
#      #X[i,2] = rbeta(1,X1+a,25-X1+b)
#    #}
#    #return(X)
#  #}
#  X0 = c(0,0.5)
#  N = 10
#  m = 5
#  a = b =1
#  #dir_cpp = 'D:/HERE/'
#  #sourceCpp(paste0(dir_cpp,"Gibbs.cpp"))
#  
#  #GibbsR = Gibbs2(N, X0)
#  #GibbsC = Gibbs1(N,m)
#  #qqplot(GibbsR[,1],GibbsC[,1])
#  #abline(a=0,b=1,col='black')
#  #qqplot(GibbsR[,2],GibbsC[,2])
#  #abline(a=0,b=1,col='black')
#  
#  #(time = microbenchmark(Gibbs1(N,m),Gibbs2(N, X0)))
#  

