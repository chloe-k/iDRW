library(igraph)
library(Matrix)
library(RWeka)
library(samr)
sapply(file.path("utils",list.files("utils", pattern="*.R")),source)

year <- 3
gene_delim <- c('g', 'm')

# make data directory
datapath <- file.path('data')
if(!dir.exists(datapath)) dir.create(datapath)

# read all data
data_all_path <- file.path(datapath, "data.RData")
if(!file.exists(data_all_path)) read_data(year, datapath)
load(data_all_path)

# global directed pathway graph provided in DRWPClass
load(file.path(datapath, "directGraph.rda"))

# Set of pathways provided in DRWPClassGM
load(file.path(datapath, "pathSet.rda"))

# directed pathway graph provided in DRWPClass
g <- directGraph 
V(g)$name <- paste("g",V(g)$name,sep="")

m <- directGraph
V(m)$name <-paste("m",V(m)$name,sep="")

r <- directGraph
V(r)$name <-paste("r",V(r)$name,sep="")

y=list(good_samples, poor_samples)

#-------- DRW-based pathway profile on a single type of feature data
# RNA-seq pathway profile
res_rna <- fit.iDRWPClass(x=list(rnaseq), y=y, globalGraph=g,
                          testStatistic= c('DESeq2'), profile_name = c('rna'),
                          datapath = datapath, pathSet=pathSet,
                          method = "DRW", samples = samples, pranking = "t-test",
                          iter = 10, DEBUG=TRUE)

# Methylation pathway profile
res_meth <- fit.iDRWPClass(x=list(imputed_methyl), y=y, globalGraph=m,
                          testStatistic= c('t-test'), profile_name = c('meth'),
                          datapath = datapath, pathSet=pathSet,
                          method = "DRW", samples = samples, pranking = "t-test",
                          iter = 10, DEBUG=TRUE)

#-------- integrative DRW on combined feature data
# concat directed pathway graphs within each profile
gm <- g %du% m
gmr <- g %du% m %du% r

testStatistic <- c("DESeq2", "t-test")
profile_name <- c("rna", "meth")

x=list(rnaseq, imputed_methyl) 

# iDRW : RNA-seq + methylation profiles (all overlapping genes)
res_rna_meth <- fit.iDRWPClass(x=x, y=y, globalGraph=gm,
                           testStatistic= testStatistic, profile_name = profile_name,
                           datapath = datapath, pathSet=pathSet,
                           method = "DRW", samples = samples, pranking = "t-test",
                           iter = 10, AntiCorr=FALSE, DEBUG=TRUE)

# iDRW-anti : RNA-seq + methylation profiles (anti-correlated genes)
res_rna_meth_anticorr <- fit.iDRWPClass(x=x, y=y, globalGraph=gm,
                                        testStatistic= testStatistic, profile_name = profile_name,
                                        datapath = datapath, pathSet=pathSet,
                                        method = "DRW", samples = samples, pranking = "t-test",
                                        iter = 10, AntiCorr=TRUE, DEBUG=TRUE)

# iDRW+DA :  RNA-seq + methylation profiles (features ranked by DA)
# da_weight_file <- "200_compressed_data.tsv"
# res_rna_meth_DA <- fit.iDRW_DA(y=y, profile_name = profile_name,
#                                datapath = datapath, pranking = "DA", 
#                                da_weight_file = da_weight_file,
#                                iter=10, DEBUG = TRUE)

#-------- comparable methods
# means / medians of the expression values of the significant pathway member genes
res_rna_meth_mean <- fit.iDRWPClass(x=x, y=y, testStatistic= testStatistic, profile_name = profile_name,
                                    datapath = datapath, pathSet=pathSet,
                                    method = "mean", samples = samples, pranking = "t-test",
                                    iter = 10, AntiCorr=FALSE, DEBUG=TRUE)

res_rna_meth_median <- fit.iDRWPClass(x=x, y=y, testStatistic= testStatistic, profile_name = profile_name,
                                      datapath = datapath, pathSet=pathSet,
                                      method = "median", samples = samples, pranking = "t-test",
                                      iter = 10, AntiCorr=FALSE, DEBUG=TRUE)

# DRW-concat : concat each pathway profile obtained from DRW
res_rna_meth_concat <- fit.iDRWconcat(y=y, profile_name = profile_name,
                                      datapath = datapath, gene_delim = gene_delim, iter = 10)

# PAC (Pathway Activities Classification) 
# source('run.pac.R')
# res_rna_meth_pac <- run.pac(xG=rnaseq,
#                             xM=imputed_methyl,
#                             y=samples)

#-------- gene profile on a single type of feature data
numFeats = 279
profile_name <- c('gf_rna', 'gf_meth')
intersect_genes <- intersect(substring(rownames(rnaseq),2), substring(rownames(imputed_methyl),2))
set.seed(1)
Feats <- intersect_genes[sample(c(1:length(intersect_genes)), numFeats)]

# RNA-seq gene profile
res_gf_rna <- fit.GM(x=list(rnaseq), y=y, testStatistic= c('DESeq2'), profile_name = c('gf_rna'),
                     feats = Feats, gene_delim = c('g'), datapath=datapath, iter=10)

# Methylation gene profile
res_gf_meth <- fit.GM(x=list(imputed_methyl), y=y, testStatistic= c('t-test'), profile_name = c('gf_meth'),
                      feats = Feats, gene_delim = c('m'), datapath=datapath, iter=10)

# concat RNA-seq + Methylation gene profile 
res_gf_rna_meth <- fit.GM(x=x, y=y, testStatistic= testStatistic, profile_name = profile_name,
                          feats = Feats, gene_delim = gene_delim, datapath=datapath, iter=10)


#-------- save and write result
# save all models
save(res_rna, res_meth, res_rna_meth, res_rna_meth_anticorr, res_rna_meth_DA,
     res_rna_meth_concat, res_rna_meth_mean, res_rna_meth_median, 
     res_gf_rna, res_gf_meth, res_gf_rna_meth,
     file=file.path(datapath,'res_models.RData'))

# write 5-fold cross validation AUC / Accuracy
res_AUC <- data.frame(Methyl=res_gf_meth[[2]], RNAseq=res_gf_rna[[2]],
                      Methyl_DRW=res_meth[[2]], RNAseq_DRW=res_rna[[2]])

res_Accuracy <- data.frame(Methyl=res_gf_meth[[3]], RNAseq=res_gf_rna[[3]],
                           Methyl_DRW=res_meth[[3]], RNAseq_DRW=res_rna[[3]])

write.table(t(res_AUC), file = paste(datapath,'result_DRW_AUC',pathend,sep=""), quote = F, col.names=F)
write.table(t(res_Accuracy), file = paste(datapath,'result_DRW_Accuracy',pathend,sep=""), quote = F, col.names=F)

# plot_perf(datapath, pathend, AUCmin=0.5, AUCmax=0.9, Accmin=50, Accmax=90)

source('graph_func.R')

perf_auc <- read.table(paste(datapath,"result_DRW_AUC",pathend,sep=""), row.names=1)
perf_acc <- read.table(paste(datapath,"result_DRW_Accuracy",pathend,sep=""), row.names=1)

row.names(perf_auc) <- c("Genes(Meth)", "Genes(Exp)", "Pathways(Meth)", "Pathways(Exp)")
row.names(perf_acc) <- c("Genes(Meth)", "Genes(Exp)", "Pathways(Meth)", "Pathways(Exp)")

g1 <- plotPerf(perf_auc, title = "", measure='mean AUC', perf_min=0.5, perf_max=0.9)

g2 <- plotPerf(perf_acc, title = "", measure='mean Accuracy(%)', perf_min=50, perf_max=90)

multiplot(g1, g2, cols=2)

#-------------------------------------------

res_AUC <- data.frame(RNAseq_Methyl=res_gf_rnaseq_methyl[[2]], 
                      RNAseq_Methyl_Mean=res_rna_meth_mean[[2]],
                      RNAseq_Methyl_Median=res_rna_meth_median[[2]],
                      RNAseq_Methyl_DRWconcat=res_rna_meth_concat[[2]],
                      RNAseq_Methyl_DRW=res_rna_meth[[2]], 
                      RNAseq_Methyl_DRWanticorr=res_rna_meth_anticorr[[2]],
                      RNAseq_Methyl_DRW_SDAE=res_rna_meth_SDAE[[2]])

res_Accuracy <- data.frame(RNAseq_Methyl=res_gf_rnaseq_methyl[[3]],
                           RNAseq_Methyl_Mean=res_rna_meth_mean[[3]],
                           RNAseq_Methyl_Median=res_rna_meth_median[[3]],
                           RNAseq_Methyl_DRWconcat=res_rna_meth_concat[[3]],
                           RNAseq_Methyl_DRW=res_rna_meth[[3]], 
                           RNAseq_Methyl_DRWanticorr=res_rna_meth_anticorr[[3]],
                           RNAseq_Methyl_DRW_SDAE=res_rna_meth_SDAE[[3]])

write.table(t(res_AUC), file = paste(datapath,'result_DRW_AUC',pathend,sep=""), quote = F, col.names=F)
write.table(t(res_Accuracy), file = paste(datapath,'result_DRW_Accuracy',pathend,sep=""), quote = F, col.names=F)

# plot_perf(datapath, pathend, AUCmin=0.5, AUCmax=0.9, Accmin=50, Accmax=90)

source('graph_func.R')

perf_auc <- read.table(paste(datapath,"result_DRW_AUC",pathend,sep=""), row.names=1)
perf_acc <- read.table(paste(datapath,"result_DRW_Accuracy",pathend,sep=""), row.names=1)

base_auc <- mean(as.numeric(perf_auc[1,]))
base_acc <- mean(as.numeric(perf_acc[1,]))

perf_auc <- perf_auc[-1,]
perf_acc <- perf_acc[-1,]

row.names(perf_auc) <- c("Mean", "Median", "DRW-concat", "iDRW", "iDRW-anti", "iDRW + DA")
row.names(perf_acc) <- c("Mean", "Median", "DRW-concat", "iDRW", "iDRW-anti", "iDRW + DA")

g1 <- plotPerf(perf_auc, title = "", measure='mean AUC', perf_min=0.5, perf_max=0.9)
g1 <- g1 + geom_hline(aes(yintercept=base_auc), linetype="dashed")

g2 <- plotPerf(perf_acc, title = "", measure='mean Accuracy(%)', perf_min=50, perf_max=90)
g2 <- g2 + geom_hline(aes(yintercept=base_acc), linetype="dashed")

multiplot(g1, g2, cols=2)


#--------- significant pathway feature selection ---------
source('func.SigFeatures.R')
pathway_df <- count.sigFeatures(res_rna, iter=10, nFolds=5)
write.SigFeatures(res_fit=res_rna, 
                  p=unlist(pathway_df[1]), 
                  pathway_cnt=as.data.frame(pathway_df[2]), 
                  profile_name="RNAseq",
                  datapath = datapath, pathend = pathend)

pathway_df <- count.sigFeatures(res_meth, iter=10, nFolds=5)
write.SigFeatures(res_fit=res_meth, 
                  p=unlist(pathway_df[1]), 
                  pathway_cnt=as.data.frame(pathway_df[2]), 
                  profile_name="Methyl",
                  datapath = datapath, pathend = pathend)

pathway_df <- count.sigFeatures(res_rna_meth, iter=10, nFolds=5)
write.SigFeatures(res_fit=res_rna_meth, 
                  p=unlist(pathway_df[1]), 
                  pathway_cnt=as.data.frame(pathway_df[2]), 
                  profile_name="RNAseq_Methyl",
                  datapath = datapath, pathend = pathend)

pathway_df <- count.sigFeatures(res_rna_meth_anticorr, iter=10, nFolds=5)
write.SigFeatures(res_fit=res_rna_meth_anticorr, 
                  p=unlist(pathway_df[1]), 
                  pathway_cnt=as.data.frame(pathway_df[2]), 
                  profile_name="RNAseq_Methyl_anticorr",
                  datapath = datapath, pathend = pathend)

pathway_df <- count.sigFeatures(res_rna_meth_SDAE, iter=10, nFolds=5)
write.SigFeatures(res_fit=res_rna_meth_SDAE, 
                  p=unlist(pathway_df[1]), 
                  pathway_cnt=as.data.frame(pathway_df[2]), 
                  profile_name="RNAseq_Methyl_DA",
                  datapath = datapath, pathend = pathend)



# prediction
# predict.DRWPClassGM(object=res_rna, 
#                     newx=rnaseq, 
#                     type = "class")

# evaluate classification performance
# evaluate.DRWPClassGM(object=fit, 
#                      newx=rnaseq, 
#                      newy.class1=good_samples, 
#                      newy.class2=poor_samples)

