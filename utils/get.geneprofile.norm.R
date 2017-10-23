get.geneprofile.norm <- function(x) {
  
  # normalize to have zero mean and unit variance 
  x.mu <- apply(x,1,mean)
  x.sigma <- apply(x,1,sd)	
  x_norm <- sweep(x,1,x.mu)
  x_norm <- sweep(x_norm,1,1/x.sigma,'*')
  
  return(x_norm)
}