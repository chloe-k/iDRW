getPathActivity <-
function(x, pathSet, w, vertexZP, method = "DRW", fname, rows){
  
	# infer pathway expression profile
  
	pathActivity <- c()
	sigGenes <- vector("list", length = length(pathSet))  # The differential genes to construct pathActivity
	names(sigGenes) <- names(pathSet)
	TValue.pathActivity <- matrix(NA, nrow = length(pathSet), ncol = 1)
	rownames(TValue.pathActivity) <- names(pathSet)
	
	if(method == "DRW") { 
	  # vertex is weighted with DRW method
	  for (i in 1 : length(pathSet)){
	    Vpathwayi <- pathSet[[i]]
	    if (length(Vpathwayi) > 0){
	      n <- 0    # the number of differential genes in ith pathway 			
	      pathActivity_tmp <- matrix(nrow=1,ncol=dim(x)[2],data=0) 
	      TValue.pathActivity_tmp <- 0
	      sigGenesi <- c()
	      Idx_pathwayi <- c()   
	      for (j in 1 : length(Vpathwayi)){
	        Idx <- which(substring(rownames(x),2)==Vpathwayi[j])
	        if (length(Idx) > 0){
	          if ( rownames(x)[Idx] %in% names(w)){
	            idx <- which(vertexZP[rownames(x)[Idx],"pvalue"] < 0.05)
	            if(length(idx) > 0){
	              for (k in 1:length(idx)) {
	                pathActivity_tmp <- pathActivity_tmp + vertexZP[rownames(x)[Idx[k]],"score"] * w[rownames(x)[Idx[k]]] * x[Idx[k],]	
	                n <- n + 1
	                Idx_pathwayi <- rbind(Idx_pathwayi,Idx[k])	
	                sigGenesi <- c(sigGenesi, rownames(x)[Idx[k]])
	              }
	            }
	          }
	        }
	      }
	      if(n > 0){
	        pathActivity_tmp <- pathActivity_tmp / sqrt(sum(w[rownames(x)[Idx_pathwayi]]^2))
	        rownames(pathActivity_tmp) <- names(pathSet)[i] 
	        pathActivity <- rbind(pathActivity, pathActivity_tmp)
	        sigGenes[[i]] <- sigGenesi
	      }
	    }
	  }
	} else {
	  # mean or median of the expression values of pathway member genes 
	  for (i in 1 : length(pathSet)){
	    Vpathwayi <- pathSet[[i]]
	    if (length(Vpathwayi) > 0){
	      n <- 0    # the number of differential genes in ith pathway 			
	      pathActivity_tmp <- c() 
	      TValue.pathActivity_tmp <- 0
	      sigGenesi <- c()
	      Idx_pathwayi <- c()   
	      for (j in 1 : length(Vpathwayi)){
	        Idx <- which(substring(rownames(x),2)==Vpathwayi[j])
	        if (length(Idx) > 0){
	          idx <- which(vertexZP[rownames(x)[Idx],"pvalue"] < 0.05)
	          if(length(idx) > 0){
	            for (k in 1:length(idx)) {
	              pathActivity_tmp <- rbind(pathActivity_tmp,x[Idx[k],])
	              n <- n + 1
	              Idx_pathwayi <- rbind(Idx_pathwayi,Idx[k])	
	              sigGenesi <- c(sigGenesi, rownames(x)[Idx[k]])
	            }
	          }
	          
	        }
	      }
	      if(n > 0){
	        names <- colnames(pathActivity_tmp)
	        if(method == "mean") {
	          pathActivity_tmp <- colMeans(pathActivity_tmp, na.rm=T)
	          
	        } else if(method == "median") {
	          
	          pathActivity_tmp <- colMedians(as.matrix(pathActivity_tmp), na.rm=T)
	        }
	        pathActivity_tmp <- matrix(pathActivity_tmp, ncol=length(pathActivity_tmp), dimnames = list(names(pathSet)[i], names))
	        
	        pathActivity <- rbind(pathActivity, pathActivity_tmp)
	        sigGenes[[i]] <- sigGenesi
	      }
	    }
	  }
	}
	
	colnames(pathActivity) <- colnames(x)
	
	# save pathway profile 
	write.table(x=t(pathActivity), file=fname, sep="\t", row.names=rows, col.names=T)
	
	return(list(pathActivity=pathActivity, sigGenes=sigGenes))	
}
