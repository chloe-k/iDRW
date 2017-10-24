fit.iDRWPClass <-
function(x, y, testStatistic, profile_name, globalGraph = NULL, datapath, pathSet,
         method = "DRW", samples, pranking = "t-test",
         classifier = "Logistic", nFolds = 5, numTops=50, 
         iter = 1, Gamma=0.7, AntiCorr = FALSE, DEBUG=FALSE) {
  
  x_norm <- list(0)
  x_stats <- list(0)
  gene_weight <- list(0)
  
  for(i in 1:length(x)) {
    # normalize gene profile
    x_norm[[i]] <- get.geneprofile.norm(x[[i]])
    
    # statistics for genes
    x_stats[[i]] <- get.genes.stats(x=x[[i]], x_norm=x_norm[[i]], y=y, 
                              DEBUG=DEBUG, testStatistic=testStatistic[i], pname=profile_name[i], datapath=datapath)
    # initialize gene weights
    geneWeight <- -log(x_stats[[i]][,2]+2.2e-16)
    geneWeight[which(is.na(geneWeight))] <- 0
    gene_weight[[i]] <- (geneWeight - min(geneWeight)) / (max(geneWeight) - min(geneWeight))
    
  }
  
  if(method == "DRW") {
    # assign initial weights to the pathway graph
    W0 <- getW0(gene_weight, globalGraph)

    if(DEBUG) cat('Performing directed random walk...')
    # get adjacency matrix of the (integrated) gene-gene graph
    W = getW(G = globalGraph, gene_weight = gene_weight, x = x_norm, datapath = datapath, EdgeWeight=FALSE, AntiCorr=AntiCorr)

    # perform DRW on gene-gene graph
    vertexWeight <- DRW(W = W, p0 = W0, gamma = Gamma)
    names(vertexWeight) <- names(W0)
    if(DEBUG) cat('Done\n')	
  } else {
    vertexWeight <- NULL
  }
	
  # reduce list of profile matrices
  x <- Reduce(rbind, x_norm)
  x_stats <- Reduce(rbind, x_stats)
  
  # pathway activity inference method
  # method = DRW / mean / median
  desc <- c(profile_name, method, if(AntiCorr) "anticorr", "txt")
  fname_profile = file.path(datapath, paste(c("pathway_profile", desc), collapse = '.'))
  
  pA <- getPathActivity(x = x, pathSet = pathSet, w = vertexWeight, vertexZP = x_stats, 
                        method = method, fname = fname_profile, rows = samples)
  
  save(pA, file=file.path(datapath, paste(c("pA", profile_name, method, if(AntiCorr) "anticorr", "RData"), collapse = '.')))
  # rank pathway activities
  # ranking = t-test / DA
  fname_rank = file.path(datapath, paste(c("pathway_rank", pranking, desc), collapse = '.'))
  
  stats_pathway <- rankPathActivity(pathActivity = pA$pathActivity, y = y, 
                                    ranking = pranking, fname=fname_rank)
	
  # perform 5-fold cross validation on logistic regression model
  respath <- "result"
  if(!dir.exists(respath)) dir.create(respath)
  
  fname_res <- file.path(respath, paste(c("result", desc), collapse = '.'))
	
  return(crossval(profile=pA$pathActivity, stats_profile=stats_pathway, y=y, sigGenes=pA$sigGenes,
           classifier=classifier, iter=iter, nFolds=nFolds, numTops=numTops, DEBUG = DEBUG, fname = fname_res))
  
}
