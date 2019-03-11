library(ggplot2)

#pathway_profile_RNAseq_drw_compressed_data.tsv - epoch 1700
#pathway_profile_Methyl_drw_compressed_data.tsv - epoch 2000
#pathway_profile_RNAseq_Methyl_drw_compressed_data.tsv - epoch 2000

file_name <- "pathway_profile_Methyl_drw_compressed_data.tsv" 
con = file(file_name, open="r")
flag = 0
n=0
linn <- readLines(con)
loss <- list()
x <- list()
for(i in 1:length(linn)){
  if(flag == 1){
    loss[n] <- linn[i]
    n <- n+1
  }
  if(linn[i] =="reconstruction error"){
    flag = 1
  }
}


close(con)

for(i in 1:(n-1)){
  x <- append(x, i)
}


max_x <- max(sapply(x, max))
min_x <- min(sapply(x, min))
max_y <- max(sapply(loss, max))
min_y <- min(sapply(loss, min))
max_y <- as.numeric(max_y)
min_y <- as.numeric(min_y)

x_axis <- as.vector(x)
y_axis <- as.vector(loss)

par(mar=c(5,5,4,2)+0.1)
plot(x_axis,y_axis, main=file_name, xlab="iteration", ylab="loss", xlim=c(min_x, max_x), ylim=c(min_y, max_y))
dev.off()
