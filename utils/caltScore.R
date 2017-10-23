caltScore <-
function(x, normSample, diseaseSample){
  
	tscore <- matrix(NA, nrow = nrow(x), ncol = 2)
	rownames(tscore) <- rownames(x)
	colnames(tscore) <- c("stat","pvalue")
	for ( i in 1 : nrow(x)){
	    tscore_tmp <- t.test(x[i,diseaseSample], x[i,normSample], var.equal=TRUE)
		tscore[i,1] <- tscore_tmp$statistic
		tscore[i,2] <- tscore_tmp$p.value
	}
	tscore <- cbind(tscore, score=sign(tscore[,1]))
	return(tscore)
}
