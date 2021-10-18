library(mice)

args <- commandArgs(trailingOnly=TRUE)
# infile = args[1]
# outfile = args[2]
infile = "/Users/sanderlindberg/Documents/TDT05/tdt05-2021-challenge-1/train.csv"
outfile = "/Users/sanderlindberg/Documents/TDT05/tdt05-2021-challenge-1/train_imputed.csv"

data_with_nan <- read.csv(infile, sep=",")
data <- complete(mice(data_with_nan, printFlag=FALSE))

write.csv(data, file=outfile, row.names=FALSE, col.names=FALSE, sep=",")
