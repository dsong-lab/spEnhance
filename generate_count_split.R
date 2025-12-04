.libPaths(Sys.getenv("R_LIBS_USER"))
print(.libPaths())

if (!requireNamespace("countsplit", quietly = TRUE)) {
  install.packages("countsplit", repos = "http://cran.us.r-project.org")
}

library(countsplit)

args <- commandArgs(trailingOnly = TRUE)
prefix <- args[1]
train_data_name <- args[2]
val_data_name <- args[3]
seed <- as.numeric(args[4])

path_to_data <- sprintf("%s/cnts.csv", prefix)
cnts <- as.matrix(read.csv(path_to_data, row.names=1))

i <- seed

set.seed(i)
cnts_split <- countsplit(cnts)
path_to_train_out <- sprintf("%s/%s_seed_%i.csv", prefix, train_data_name, i)
write.csv(as.data.frame(as.matrix(cnts_split[[1]])), 
          file = path_to_train_out, 
          row.names = TRUE)
path_to_test_out <- sprintf("%s/%s_seed_%i.csv", prefix, val_data_name, i)
write.csv(as.data.frame(as.matrix(cnts_split[[2]])), 
          file = path_to_test_out, 
          row.names = TRUE)

