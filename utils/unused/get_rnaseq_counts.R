biocLite("RTCGAToolbox")

library(RTCGAToolbox)

brcaData = getFirehoseData(dataset = "BRCA", 
                           runDate=getFirehoseRunningDates()[1],
                           Clinic = T,
                           RNAseq_Gene = T,
                           miRNASeq_Gene = T,
                           RNAseqNorm = "raw_counts")

getData(brcaData, "")
rnaseqCounts = getData(brcaData, "RNASeqGene")
mirnaseqCounts = getData(brcaData, "miRNASeqGene")
head(rnaseqCounts)
head(mirnaseqCounts)

colnames(rnaseqCounts) <- gsub('-','.',substring(colnames(rnaseqCounts),1,15))
colnames(mirnaseqCounts) <- gsub('-','.',substring(colnames(mirnaseqCounts),1,15))
head(rnaseqCounts)
head(mirnaseqCounts)

summary((colnames(rnaseq[,2:length(colnames(rnaseq))]) %in% colnames(rnaseqCounts)) | (colnames(rnaseq[,2:length(colnames(rnaseq))]) %in% colnames(mirnaseqCounts)))



rownames(clinical) %in% colnames(rnaseqCounts)
rownames(clinical) %in% colnames(mirnaseqCounts)


Clinicaldata = getData(brcaData, "Clinical")
identical(tolower(substring(rownames(clinical),1,12)),rownames(Clinicaldata))
