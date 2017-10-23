fitModel_geneProfile <-
  function(expr_training, expr_test, classifier = "Logistic", numTops = 50) {
    res.fitModel <- fitModel(expr_training, expr_test, classifier)
    eString <- unlist(strsplit(res.fitModel$eTestSet$string,"\n"))
    eString.4 <- unlist(strsplit(eString[4], "[[:blank:]]+"))
    Accuracy <- as.numeric(eString.4[5])
    eString.20 <- unlist(strsplit(eString[18],"[[:blank:]]+"))
    AUC <- as.numeric(eString.20[9])

    return(list(model=res.fitModel, AUC=AUC, Accuracy=Accuracy))
  }
