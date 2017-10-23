rankPathActivity <- function(pathActivity=NULL, y=NULL, ranking = "t-test", fname, DApath=NULL){
  
  if(ranking == "t-test") {
    
    # rank pathway activities by t-test statistics
    statsPA <- caltScore(pathActivity, y[[1]], y[[2]])
    stats_pathway <- statsPA[ ,1]
    Idx <- sort(stats_pathway,decreasing = TRUE,index.return=TRUE)$ix
    stats_pathway <- stats_pathway[Idx]
    
  } else if(ranking == "DA") {
    # rank pathway activities by weight matrix of DA
    stats_pathway <- read.delim(file = DApath, header = F,row.names = 1,col.names = c("", "weight"))
    stats_pathway <- apply(t(stats_pathway),2,as.numeric)
    
    library(stringr)
    names(stats_pathway) <- str_pad(names(stats_pathway), 5, pad = "0")
  }
  
  # write pathway ranking
  write.table(x=matrix(t(stats_pathway), ncol=length(stats_pathway), dimnames=list("rank",names(stats_pathway))),
              file=fname, sep="\t", row.names=T, col.names=T)
  
  return(stats_pathway)
}