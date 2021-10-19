library(mice)

args <- commandArgs(trailingOnly=TRUE)
infile = args[1]
outfile = args[2]

data_with_nan <- read.csv(infile, sep=",")
data <- complete(mice(data_with_nan, printFlag=FALSE, method = "cart"))

write.csv(data, file=outfile, row.names=FALSE, col.names=FALSE, sep=",")
