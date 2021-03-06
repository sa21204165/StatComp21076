% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/StatComp.R
\name{WSVT}
\alias{WSVT}
\title{WSVT}
\usage{
WSVT(M, w, lamda)
}
\arguments{
\item{M}{sparse matrix}

\item{w}{weight}

\item{lamda}{tunning parameter}
}
\value{
complete matrix
}
\description{
Nonconvex Nonsmooth Low Rank Minimization via Iteratively Reweighted Nuclear Norm
}
\examples{
\dontrun{
M <- matrix(c(0,1,0,1,0,1,0,1,0),3,3)
g <- SVP(M,1,0.01,2,10000)
}
}
