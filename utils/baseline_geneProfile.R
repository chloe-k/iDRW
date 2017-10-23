baseline_geneProfile <- function(rnaseq, imputed_methyl, 
                               good_samples, poor_samples, normalize = T,
                               numTops = 50, numFeats = 279, iter = 1, datapath, pathend) {
  
  numFeats_idxG <- sample(c(1:nrow(rnaseq)), numFeats)
  numFeats_idxM <- sample(c(1:nrow(imputed_methyl)), numFeats)
  
  res_gf_rnaseq <- fit.GM(xG = rnaseq, 
                          yG.class1 = good_samples, 
                          yG.class2 = poor_samples, 
                          classifier = "Logistic", 
                          testStatistic = "DESeq2",
                          normalize = T,
                          profile_name = "RNAseq",
                          numTops = numTops, numFeats_idxG = numFeats_idxG,
                          iter=iter, datapath=datapath, pathend=pathend)
  
  
  res_gf_methyl <- fit.GM(xG = imputed_methyl, 
                          yG.class1 = good_samples, 
                          yG.class2 = poor_samples, 
                          classifier = "Logistic", 
                          testStatistic = "t-test",
                          normalize = T,
                          profile_name = "Methyl",
                          numTops = numTops, numFeats_idxG = numFeats_idxM,
                          iter=iter, datapath=datapath, pathend=pathend)
  
  
  res_gf_rnaseq_methyl <- fit.GM(xG = rnaseq, xM = imputed_methyl,
                                 yG.class1 = good_samples, 
                                 yG.class2 = poor_samples, 
                                 classifier = "Logistic", 
                                 testStatistic = "DESeq2",
                                 normalize = T,
                                 profile_name = "RNAseq_Methyl",
                                 numTops = numTops, numFeats_idxG = numFeats_idxG, 
                                 numFeats_idxM = numFeats_idxM,
                                 iter=iter, datapath=datapath, pathend=pathend)
  
  save(res_gf_rnaseq, res_gf_methyl, res_gf_rnaseq_methyl, file=paste('Data/res_models_gf.DESeq2.',year,'y.iter10.RData',sep=""))
}

