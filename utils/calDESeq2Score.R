calDESeq2Score <-
  function(x, normSample, diseaseSample){

    library(DESeq2)
    
    ret <- list()
    col.data <- data.frame(condition=rep("GOOD", length(normSample)), row.names = colnames(x[,normSample]))
    col.data <- rbind(col.data, data.frame(condition=rep("POOR", length(diseaseSample)), row.names=colnames(x[,diseaseSample])))
    col.data$condition <- factor(col.data$condition, levels = c("GOOD", "POOR"))
    
    count.matrix <- cbind(x[,normSample], x[,diseaseSample])
    
    #print(col.data)
    ckCDS <- DESeqDataSetFromMatrix(countData = round(count.matrix),
                                            colData = col.data,
                                            design = ~condition)
    
    dds <-DESeq(ckCDS)
    
    res <- as.matrix(results(dds))
    
    res <- res[,c(4,5,2)]
    colnames(res)[3]<-"score"
    
    return(res)
  }