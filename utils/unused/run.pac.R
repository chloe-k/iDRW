run.pac <-
  function(xG, xM, y, normalize=T, AntiCorr=F) {

    library(netClass)
    library(tdROC)


    source('pacCV.R')

    data(expr)
    data(ad.matrix)
    #
    library(KEGG.db)
    # # x (p=135 samples x n=1000 genes), numeric matrix
    # # y (class labels on p=135 samples), numeric factor
    # # Gsub (adjacency matrix of gene-gene graph)

    #source("get.geneprofile.norm.R")
    #source("get.genes.stats.R")

    res <- get.geneprofile.norm(xG, xM)
    xG_norm <- res[[1]]
    xM_norm <- res[[2]]

    res <- get.genes.stats(xG, xG_norm, yG.class1, yG.class2,
                           xM_norm=NULL, yM.class1=NULL, yM.class2=NULL,
                           DEBUG=TRUE, year,
                           profile_name="RNAseq_Methyl", testStatistic="DESeq2")
    statsG <- res[[1]]
    statsM <- res[[2]]

    if(AntiCorr==FALSE){
      fname <- paste("Data/W_all_overlapping_edges.",year,"y.RData",sep="") # 88440 edges
    } else{
      fname <- paste("Data/W_anti_overlapping_edges_0.5.",year,"y.RData", sep="") # 87988 edges
    }
    if(file.exists(fname)) {
      load(fname)
    }

    x <- t(as.matrix(rbind(xG_norm, xM_norm)))
    y[good_samples] <- 1
    y[poor_samples] <- -1
    y <- as.factor(y)

    statsM <- cbind(statsM, log2FoldChange = sign(statsM[,"stat"]))
    stats <- rbind(statsG, statsM)


    n     <- length(y)  # 465 samples
    folds <- trunc(10)
    inter= intersect(colnames(W), colnames(x))  # 6944 genes in the gene-gene graph
    W = W[inter,inter]	# 6944 x 6944
    x=x[,inter]  # 465 x 6944
    stats=stats[inter,]


    res <- probeset2pathwayTrain(x=x, y=y, int=inter)

    # r.pac <- cv.pac(x=t(x), y=y,
    #                 folds=5, repeats=10,
    #                 parallel=FALSE, cores=2, DEBUG=TRUE,
    #                 Gsub=W, seed=1234)
  }

