plot_perf <- function(datapath, pathend, AUCmin, AUCmax, Accmin, Accmax) {
  source('graph_func.R')
  
  # bar plot
  
  perf_auc <- read.table(paste(datapath,"result_DRW_AUC",pathend,sep=""), row.names=1)
  perf_acc <- read.table(paste(datapath,"result_DRW_Accuracy",pathend,sep=""), row.names=1)
  
  g1 <- plotPerf(perf_auc, title = "", measure='mean AUC', perf_min=AUCmin, perf_max=AUCmax)
  g2 <- plotPerf(perf_acc, title = "", measure='mean Accuracy', perf_min=Accmin, perf_max=Accmax)
  
  multiplot(g1, g2, cols=2)
  
}




