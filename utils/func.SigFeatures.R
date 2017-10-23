count.sigFeatures <- function(res_fit, iter, nFolds) {
  # union significant pathway / gene features across 5 folds
  res <- res_fit[[4]]
  r <- res[[2]]
  p <- names(r)
  for (i in 2:iter*nFolds) {
    res <- res[[1]]
    r <- res[[2]]
    p <- union(p,names(r))
  }
  pathway_cnt <- data.frame(pathway_id=p, count=rep(0,length(p)), row.names = 1)
  
  # count significant pathway / gene features across 5 folds
  res <- res_fit[[4]]
  r <- res[[2]]
  pathway_cnt[names(r),] <- pathway_cnt[names(r),] + 1
  for (i in 2:iter*nFolds) {
    res <- res[[1]]
    r <- res[[2]]
    pathway_cnt[names(r),] <- pathway_cnt[names(r),] + 1
  }
  return(list(p, pathway_cnt))
}

# write significant pathway / gene features
write.SigFeatures <- function(res_fit, p, pathway_cnt, profile_name, datapath, pathend) {
  
  library(KEGG.db)
  pathway_name <- mget(p,KEGGPATHID2NAME,ifnotfound=NA)
  
  genemap <- get.geneMapTable()
  sink(paste(datapath,"sigPathway_genes_",profile_name,pathend,sep="", collapse=NULL))
  for (i in 1:length(p)) {
    sigGeneSet <- res_fit[[5]]
    sigGenes <- unlist(sigGeneSet[p[i]],use.names = F)
    cat(p[i], ';', as.character(pathway_name[p[i]]), ';', pathway_cnt[p[i],], ';', geneid2symbol(substring(sigGenes,2),genemap,substring(sigGenes,1,1)), '\n')
  }
  sink()
}

geneid2symbol <- function(gene, genemap, gid) {
  symbol <- as.character(genemap[gene,])
  for (i in 1:length(symbol)) {
    if(gid[i]=="g") symbol[i] <- paste(symbol[i], "(RNAseq)",sep="")
    else if(gid[i]=="m") symbol[i] <- paste(symbol[i], "(Methyl)", sep="")
    
  }
  return(symbol)
}

get.geneMapTable <- function() {
  genemap <- read.table('../Data/BRCA_rnaseq_methylation/gene_name_id_map', sep = ',', row.names=2)
  return(genemap)
  }