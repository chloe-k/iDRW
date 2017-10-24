fit.GM <- function(x, y, testStatistic, profile_name, feats, gene_delim, datapath, 
                   classifier = "Logistic", nFolds = 5, numTops=50, iter = 1){
  
  x_norm <- list(0)
  x_stats <- list(0)
  gene_weight <- list(0)
  
  for(i in 1:length(x)) {
    x[[i]] <- x[[i]][paste(gene_delim[i],feats,sep=""),]
    
    # normalize gene profile
    x_norm[[i]] <- get.geneprofile.norm(x[[i]])
    
    # statistics for genes
    x_stats[[i]] <- get.genes.stats(x=x[[i]], x_norm=x_norm[[i]], y=y, 
                                    DEBUG=TRUE, testStatistic=testStatistic[i], pname=profile_name[i], datapath=datapath)
  }
  
  # reduce list of profile matrices
  x <- Reduce(rbind, x_norm)
  x_stats <- Reduce(rbind, x_stats)
  
  # rank genes by their scores
  stats_genes <- x_stats[ ,1]
  Idx <- sort(stats_genes,decreasing = TRUE,index.return=TRUE)$ix
  stats_genes <- stats_genes[Idx]
  
  
  # perform 5-fold cross validation on logistic regression model
  respath <- "result"
  if(!dir.exists(respath)) dir.create(respath)
  
  desc <- c(profile_name, "txt")
  fname_res <- file.path(respath, paste(c("result", desc), collapse = '.'))
  
  return(crossval(profile=x, stats_profile = stats_genes, y=y,
                  classifier=classifier, iter=iter, nFolds=nFolds, numTops=numTops, DEBUG = TRUE, fname = fname_res))
}
