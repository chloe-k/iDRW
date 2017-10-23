fit.iDRWconcat <- function(y, profile_name, datapath, gene_delim,
                               iter=1, nFolds=5, numTops = 50, 
                               classifier = "Logistic", 
                               method = "DRW", AntiCorr=FALSE, DEBUG = TRUE) {
  
  path_activity <- list(0)
  path_rank <- list(0)
  
  for (i in 1:length(profile_name)) {
    # read pathway profile
    desc <- c(profile_name[[i]], method, if(AntiCorr) "anticorr", "RData")
    fname_profile = file.path(datapath, paste(c("pathway_profile", desc), collapse = '.'))
    
    path_activity[[i]] <- read.table(fname_profile, header=T, check.names = F)
    names(path_activity[[i]]) <- paste(gene_delim[[i]], colnames(path_activity[[i]]), sep="")
    
    # read pathway rank
    fname_rank = file.path(datapath, paste(c("pathway_rank", desc), collapse = '.'))
    
    path_rank[[i]] <- as.data.frame(read.table(file = fname_rank, check.names = F))
    names(path_rank[[i]]) <- paste(gene_delim[[i]], colnames(path_rank[[i]]), sep="")
    path_rank[[i]] <- apply(path_rank[[i]],2,as.numeric)
  }
  
  # merge pathway profile
  cpathway_profile <- Reduce(cbind, path_activity)
  
  # merge rank of pathways
  cpathway_rank <- Reduce(c, path_rank)
  
  # perform 5-fold cross validation on logistic regression model
  respath <- "result"
  if(!dir.exists(respath)) dir.create(respath)
  
  desc <- c(profile_name, method, "concat", if(AntiCorr) "anticorr", "txt")
  fname_res <- file.path(respath, paste(c("result", desc), collapse = '.'))
  
  return(crossval(pathActivity=t(cpathway_profile), stats_pathway=cpathway_rank, y=y,
                  classifier=classifier, iter=iter, nFolds=nFolds, numTops=numTops, DEBUG = DEBUG, fname = fname_res))
}