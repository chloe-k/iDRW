# pmat <- read.csv('Data/t.test_3years/DESeq2_170609/pathways_coeff_matrix.csv', row.names = 1)
# 
# 
# library('pheatmap')
# 
# 
# myColor <- colorRampPalette(c("white", "blue"))
# p<-pheatmap(pmat, clustering_method = "complete", color=myColor(50), cluster_rows=F, cluster_cols=F, border_color = NA)
# 
# p<-pheatmap(pmat, clustering_method = "complete", color=myColor(50), border_color = NA)
# grid.text("DRW", y=-0.07, gp=gpar(fontsize=16))
# grid.text("DRW+DA", x=-0.07, rot=90, gp=gpar(fontsize=16))
