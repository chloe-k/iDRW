crossval <- function(profile, stats_profile, y, sigGenes=NULL,
                     classifier = "Logistic", iter=1, nFolds=5, numTops=50, DEBUG=FALSE, fname) {
  
  AUC <- 0
  Accuracy <- 0
  sink(fname)
  perfAUC <- c()
  perfAcc <- c()
  perfFeatures <- list()
  
  y1 <- y[[1]]
  y2 <- y[[2]]
  
  for(i in 1:iter){
    if(DEBUG) cat('iter..................................', i, '\n')
    
    for(j in 1:nFolds){
      if(DEBUG) cat('iter=', i,'Folds=', j, '\n')	
      y1.test <- sample(y1, floor(length(y1) / nFolds))
      y1.training <- setdiff(y1, y1.test)
      y2.test <- sample(y2, floor(length(y2) / nFolds))
      y2.training <- setdiff(y2, y2.test)
      
      pA.training <- profile[,c(y1.training, y2.training)]
      pA.test <- profile[,c(y1.test, y2.test)]
      
      y1.training  <- 1:length(y1.training)
      y2.training <- (1+length(y1.training)):(length(y1.training)+length(y2.training))
      y1.test  <- 1:length(y1.test)
      y2.test <- (1+length(y1.test)):(length(y1.test)+length(y2.test))
      
      classType_training <- rep(NA, dim(pA.training)[2])
      classType_training[y1.training] <- "class1"
      classType_training[y2.training] <- "class2"
      arffRW_training <- data.frame(t(pA.training), "class" = classType_training, check.names=F)
      
      classType_test <- rep(NA, dim(pA.test)[2])
      classType_test[y1.test] <- "class1"
      classType_test[y2.test] <- "class2"
      arffRW_test <- data.frame(t(pA.test), "class" = classType_test, check.names=F)
      
      resPredict <- fitModelGreedy(arffRW_training, arffRW_test, stats_profile, classifier = classifier, numTops=numTops)
      if(DEBUG){
        print(list(model=resPredict$model, AUC=resPredict$AUC, 
                   sigFeatures = resPredict$sigFeatures, geneFeatures = sigGenes[resPredict$sigFeatures]))
      }
      
      AUC <- resPredict$AUC
      Accuracy <- resPredict$Accuracy
      fit <- list(model=resPredict$model, AUC=resPredict$AUC, Accuracy=resPredict$Accuracy, 
                  sigFeatures = resPredict$sigFeatures, geneFeatures = sigGenes[resPredict$sigFeatures])
      
      perfAUC <- c(perfAUC, AUC)
      perfAcc <- c(perfAcc, Accuracy)
      perfFeatures <- list(perfFeatures, sigGenes[resPredict$sigFeatures])
      
      if(resPredict$AUC > AUC){
        # print(i)
        AUC <- resPredict$AUC
        Accuracy <- resPredict$Accuracy
        fit <- list(model=resPredict$model, AUC=resPredict$AUC, Accuracy=resPredict$Accuracy, 
                    sigFeatures = resPredict$sigFeatures, geneFeatures = sigGenes[resPredict$sigFeatures])
        
      }else{
        if(resPredict$AUC == AUC & resPredict$Accuracy > Accuracy){
          # print(i)
          AUC <- resPredict$AUC
          Accuracy <- resPredict$Accuracy
          fit <- list(model=resPredict$model, AUC=resPredict$AUC, Accuracy=resPredict$Accuracy, 
                      sigFeatures = resPredict$sigFeatures, geneFeatures = sigGenes[resPredict$sigFeatures])
          
        }
      }
    }
    
  }
  sink()
  return(list(fit,perfAUC,perfAcc,perfFeatures,sigGenes))
}