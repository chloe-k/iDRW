fit.GM <-
  function(xG, xM = NULL, yG.class1, yG.class2, 
           testStatistic = "t-test",
           classifier = "Logistic", profile_name, normalize = F,
           numTops = 50, iter = 1, nFolds = 5, numFeats_idxG = numFeats_idxG, numFeats_idxM = NULL, DEBUG = TRUE,
           datapath, pathend){
    
    xG <- xG[numFeats_idxG,]
    if(!is.null(xM)) xM <- xM[numFeats_idxM,]

    cat(dim(xG), '\n')
    
    res <- get.geneprofile.norm(xG=xG, xM=xM)
    xG_norm <- res[[1]]
    xM_norm <- res[[2]]
    
    cat(dim(xG_norm), '\n')
    
    res <- get.genes.stats(xG=xG, xG_norm=xG_norm, yG.class1=yG.class1, yG.class2 = yG.class2,
                           xM_norm = xM_norm, yM.class1 = yG.class1, yM.class2 = yG.class2,
                           DEBUG=TRUE, year=year,
                           profile_name = profile_name, testStatistic=testStatistic)
    statsG <- res[[1]]
    statsM <- res[[2]]
    
    cat(dim(statsG), '\n')
    
    statsG <- statsG[numFeats_idxG,]
    if(!is.null(statsM)) statsM <- statsM[numFeats_idxM,]
    
    cat(dim(statsG), '\n')
    
    if(!is.null(xM)){
      cat('integrated')
      # merge xG and xM
      x <- rbind(xG_norm, xM_norm)
      if(testStatistic == "DESeq2") statsM <- cbind(statsM, log2FoldChange = sign(statsM[,"stat"]))
      stats <- rbind(statsG, statsM)
      
    } else{
      x <- xG_norm
      stats <- statsG
    }
    
    stats_genes <- stats[ ,1]
    
    # cat(length(stats_genes), '\n')
    # cat(dim(x), '\n')
    # stats_genes <- stats_genes[sample(c(1:length(stats_genes)), numFeats)]
    # Idx <- sort(stats_genes,decreasing = TRUE,index.return=TRUE)$ix
    # stats_genes <- stats_genes[Idx]
    # stats_genes <- stats_genes[1:numFeats]
    
    # stats <- as.matrix(stats_genes)
    # stats <- stats[order(stats$`pvalue`, decreasing=F),]
    # stats <- stats[1:numFeats,]
    # stats <- as.matrix(stats)
    
    # x <- x[names(stats_genes),]
    
    # cat(dim(x),'\n')
    # cat(length(stats_genes),'\n')
  
    profile_name = paste(profile_name, ".", testStatistic, sep="")
    
    AUC <- 0
    Accuracy <- 0
    sink(paste(datapath,"result_gf_",profile_name,pathend,sep=""))
    perfAUC <- c()
    perfAcc <- c()
    
    for(i in 1:iter){
      if(DEBUG) cat('iter..................................', i, '\n')
      for(j in 1:nFolds){
        if(DEBUG) cat('iter=', i,'Folds=', j, '\n')	
        yG.class1.test <- sample(yG.class1, floor(length(yG.class1) / nFolds))
        yG.class1.training <- setdiff(yG.class1, yG.class1.test)
        yG.class2.test <- sample(yG.class2, floor(length(yG.class2) / nFolds))
        yG.class2.training <- setdiff(yG.class2, yG.class2.test)
        
        x.training <- x[,c(yG.class1.training, yG.class2.training)]
        x.test <- x[,c(yG.class1.test, yG.class2.test)]
        
        yG.class1.training  <- 1:length(yG.class1.training)
        yG.class2.training <- (1+length(yG.class1.training)):(length(yG.class1.training)+length(yG.class2.training))
        yG.class1.test  <- 1:length(yG.class1.test)
        yG.class2.test <- (1+length(yG.class1.test)):(length(yG.class1.test)+length(yG.class2.test))
        
        classType_training <- rep(NA, dim(x.training)[2])
        classType_training[yG.class1.training] <- "class1"
        classType_training[yG.class2.training] <- "class2"
        arffRW_training <- data.frame(t(x.training), "class" = classType_training, check.names=F)
        
        classType_test <- rep(NA, dim(x.test)[2])
        classType_test[yG.class1.test] <- "class1"
        classType_test[yG.class2.test] <- "class2"
        arffRW_test <- data.frame(t(x.test), "class" = classType_test, check.names=F)
        
        # browser()
        resPredict <- fitModelGreedy(arffRW_training, arffRW_test, stats_genes, classifier = classifier, numTops=numTops)
        #resPredict <- fitModel_geneProfile(arffRW_training, arffRW_test, classifier = classifier, numTops=numTops)
        
        if(DEBUG){
          # print(i)
          print(list(model=resPredict$model, AUC=resPredict$AUC))
        }
        
        AUC <- resPredict$AUC
        Accuracy <- resPredict$Accuracy
        fit <- list(model=resPredict$model, AUC=resPredict$AUC, Accuracy=resPredict$Accuracy)
        
        perfAUC <- c(perfAUC, AUC)
        perfAcc <- c(perfAcc, Accuracy)
        
        if(resPredict$AUC > AUC){
          # print(i)
          AUC <- resPredict$AUC
          Accuracy <- resPredict$Accuracy
          fit <- list(model=resPredict$model, AUC=resPredict$AUC, Accuracy=resPredict$Accuracy)
          
        }else{
          if(resPredict$AUC == AUC & resPredict$Accuracy > Accuracy){
            # print(i)
            AUC <- resPredict$AUC
            Accuracy <- resPredict$Accuracy
            fit <- list(model=resPredict$model, AUC=resPredict$AUC, Accuracy=resPredict$Accuracy)
          }
        }
      }
    }
    sink()
    class(fit) <- "DRWPClassGM"
    return(list(fit,perfAUC,perfAcc))
  }
