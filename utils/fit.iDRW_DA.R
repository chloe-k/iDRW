fit.iDRW_DA <- function(y, profile_name, datapath, pranking, da_weight_file,
                               iter=1, nFolds=5, numTops = 50, 
                               classifier = "Logistic", 
                               method = "DRW", AntiCorr=FALSE, DEBUG = TRUE) {
  
  # read pathway profile
  fname_pA = file.path(datapath, paste(c("pA", profile_name, method, if(AntiCorr) "anticorr", "RData"), collapse = '.'))
  load(file=fname_pA)
  
  # read pathway rank from DA
  fname_rank = file.path(datapath, paste(c("pathway_rank", pranking, profile_name, method, if(AntiCorr) "anticorr", "txt"), collapse = '.'))
  DApath <- file.path(datapath, "DA_result", da_weight_file)
  pathway_rank <- rankPathActivity(ranking=pranking, fname=fname_rank, DApath=DApath)
  
  # perform 5-fold cross validation on logistic regression model
  respath <- "result"
  if(!dir.exists(respath)) dir.create(respath)
  
  desc <- c(profile_name, method, "da", if(AntiCorr) "anticorr", "txt")
  fname_res <- file.path(respath, paste(c("result", desc), collapse = '.'))
  
  return(crossval(pathActivity=pA$pathActivity, stats_pathway=pathway_rank, y=y, sigGenes=pA$sigGenes,
                  classifier=classifier, iter=iter, nFolds=nFolds, numTops=numTops, DEBUG = DEBUG, fname = fname_res))
}