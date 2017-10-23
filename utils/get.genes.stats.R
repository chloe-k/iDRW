get.genes.stats <- function(x, x_norm, y,
                      DEBUG=FALSE, testStatistic, pname, datapath) {
  
  fname = file.path(datapath, paste(testStatistic,pname,"RData",sep="."))
  
  if(!file.exists(fname)) {
    if(testStatistic == "t-test"){
      if(DEBUG) cat('Calculating t-test score...\n')
      stats <- caltScore(x_norm, y[[1]], y[[2]])
      
    } else if(testStatistic == "DESeq2"){
      if(DEBUG) cat('Calculating DESeq2 score...\n')
      stats <- calDESeq2Score(x, y[[1]], y[[2]])
      
    } 
    save(stats, file=fname)
    
  } else {
    stats <- get(load(fname))
  }
  
  if(DEBUG) cat('Done\n')
  return(stats)

}