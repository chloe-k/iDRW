getW <- function(G, gene_weight, x, datapath, EdgeWeight=FALSE, AntiCorr=FALSE) {
  
  # make graph data directory
  graphpath <- file.path(datapath, "Graph")
  if(!dir.exists(graphpath)) dir.create(graphpath)
  
  len = length(gene_weight)
  if(len > 1) {
    intersect_genes <- Reduce(intersect, lapply(gene_weight, function(x) substring(x,2)))
    
    if(!EdgeWeight) {
      if(!AntiCorr) {
        
        # assign bi-directional edges to all overlapping genes between exp & meth
        fname <- file.path(graphpath, "W_all_overlapping_edges.RData") # 88440 edges
        if(file.exists(fname)) {
          load(fname)
        } else {
          W <- as.matrix(get.adjacency(G))
          
          for(i in 1:length(intersect_genes)){
            idx=which(paste("g",intersect_genes[i],sep="")==rownames(W))
            if(length(idx)>0) {
              W[paste("g",intersect_genes[i],sep=""),paste("m",intersect_genes[i],sep="")] <- 1
              W[paste("m",intersect_genes[i],sep=""),paste("g",intersect_genes[i],sep="")] <- 1
            }
          }
          save(W, file=fname)
        }
      } else if(AntiCorr) {
        
        # assign bi-directional edges to significantly anti-correlated genes between exp & meth
        fname <- file.path(graphpath, "W_anti_overlapping_edges.RData") # 81750 edges
        if(file.exists(fname)) {
          load(fname)
        } else {
          W <- as.matrix(get.adjacency(G))
          xG <- x[[1]]
          xM <- x[[2]]
          
          for(i in 1:length(intersect_genes)){
            idx=which(paste("g",intersect_genes[i],sep="")==rownames(W))
            if(length(idx)>0) {
              if(cor(t(xG[paste("g",intersect_genes[i],sep=""),]),
                     t(xM[paste("m",intersect_genes[i],sep=""),])) < 0 &
                 cor.test(t(xG[paste("g",intersect_genes[i],sep=""),]),
                          t(xM[paste("m",intersect_genes[i],sep=""),]),
                          method = "pearson", alternative = "less")$p.value <= 0.05) {
                W[paste("g",intersect_genes[i],sep=""),paste("m",intersect_genes[i],sep="")] <- 1
                W[paste("m",intersect_genes[i],sep=""),paste("g",intersect_genes[i],sep="")] <- 1
              }
            }
          }
          save(W, file=fname)
        }
      }
    }
  } else {
    W <- as.matrix(get.adjacency(G))
  }
  
  print(dim(W)) # number of nodes
  print(sum(W)) # number of edges (adjacency matrix)
  
  print('Adjacency matrix W complete ...')
  
  return(W)
}
