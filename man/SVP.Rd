% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/StatComp.R
\name{SVP}
\alias{SVP}
\title{SVP}
\usage{
SVP(M, step_size, tolerance, rank_r, k_max)
}
\arguments{
\item{M}{sparse matrix}

\item{step_size}{step size}

\item{tolerance}{tolerance}

\item{rank_r}{Rank Minimization}

\item{k_max}{maximum number of iterations}
}
\value{
complete matrix
}
\description{
Guaranteed Rank Minimization via Singular Value Projection
}
\examples{
\dontrun{
M <- matrix(c(0,1,0,1,0,1,0,1,0),3,3)
g <- SVP(M,1,0.01,2,10000)
}
}
