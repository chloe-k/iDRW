baseline_geneProfile <- function(x, y, numTops = 50, numFeats, iter = 1, datapath) {
  
  numFeats_idxG <- sample(c(1:nrow(rnaseq)), numFeats)
  numFeats_idxM <- sample(c(1:nrow(imputed_methyl)), numFeats)
  
  
  res_gf_rnaseq <- fit.GM(x = x[[1]], y = y, 
                          testStatistic = testStatistic[1],
                          profile_name = profile_name[1],
                          numFeats = numFeats,
                          iter=iter, datapath=datapath, pathend=pathend)
  
  
  res_gf_methyl <- fit.GM(x = x[[2]], y = y, 
                          classifier = "Logistic", 
                          testStatistic = "t-test",
                          normalize = T,
                          profile_name = "Methyl",
                          numTops = numTops, numFeats_idxG = numFeats_idxM,
                          iter=iter, datapath=datapath, pathend=pathend)
  
  
  res_gf_rnaseq_methyl <- fit.GM(x = x, y = y, 
                                 classifier = "Logistic", 
                                 testStatistic = "DESeq2",
                                 normalize = T,
                                 profile_name = "RNAseq_Methyl",
                                 numTops = numTops, numFeats_idxG = numFeats_idxG, 
                                 numFeats_idxM = numFeats_idxM,
                                 iter=iter, datapath=datapath, pathend=pathend)
  
  save(res_gf_rnaseq, res_gf_methyl, res_gf_rnaseq_methyl, file=paste('Data/res_models_gf.DESeq2.',year,'y.iter10.RData',sep=""))
}

