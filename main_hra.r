# clear up everything 
rm(list=ls(all=TRUE))

# read data split function
source("G:/Kaggle/code/shared_code/data_split.R") # data split for training and testing
source("G:/Kaggle/code/shared_code/multi_plot.R") # put multiple plots into one
source("G:/Kaggle/code/shared_code/eval_clas_bnr.R") # the binary classifier evaluation code

# set the directory to get the data
setwd("G:/Kaggle/data/Human.Resources.Analytics")

# call package
library(plyr)
library(dplyr)
library(ggplot2)
library(Amelia)
library(ROCR)
library(pROC)
library(rpart)
#install.packages("randomForest")
library(randomForest)
#install.packages("neuralnet")
library(neuralnet)
library(e1071)

# read dataset
hra_data=read.csv('HR_comma_sep.csv',header=TRUE,stringsAsFactors = F)

# take a look of the structure of the dataset
str(hra_data)

# check missing value in the dataset
sapply(hra_data,function(x) sum(is.na(x)))
missmap(hra_data)

# look at the variable in the dataset

# data preparation
# split training and testing dataset
# general dataset
hra_data$sales <- factor(hra_data$sales)
hra_data$salary <- factor(hra_data$salary)

splits <- splitdf(hra_data,seed=123,ratio_training=0.7)
str(splits)

training <- splits$trainset
testing <- splits$testset
testing <- plyr::rename(testing,c("left"="real_left"))

# dummy variable
hra_data_dm <- model.matrix(~.,data=hra_data)
hra_data_dm <- hra_data_dm[,c(2:ncol(hra_data_dm))]
hra_data_dm <- as.data.frame(hra_data_dm)

splits_dm <- splitdf(hra_data_dm,seed=123,ratio_training=0.7)
str(splits_dm)

training_dm <- splits_dm$trainset
testing_dm <- splits_dm$testset
testing_dm <- plyr::rename(testing_dm,c("left"="real_left"))

# dummy variable with scaling

maxs <- apply(hra_data_dm, 2, max) 
mins <- apply(hra_data_dm, 2, min)
hra_data_dm_sc <- as.data.frame(scale(hra_data_dm, center = mins, scale = maxs - mins))

splits_dm_sc <- splitdf(hra_data_dm_sc,seed=123,ratio_training=0.7)
str(splits_dm_sc)

training_dm_sc <- splits_dm_sc$trainset
testing_dm_sc <- splits_dm_sc$testset
testing_dm_sc <- plyr::rename(testing_dm_sc,c("left"="real_left"))

# start machine learning!
# set the timer
start_time <- proc.time()

# decision tree
# hra.dt <- rpart(left~.,training,method='class')
# 
# hra.dt.pdt <- predict(hra.dt,testing,type='class')
# 
# eval_clas_bnr('decision tree',hra.dt.pdt,testing$real_left)

# logistic regression
# hra.lr <- glm(factor(left)~.,family=binomial(link='logit'),data=training)
# 
# hra.lr.pdt <- ifelse(predict(hra.lr,testing,type='response')>0.5,1,0)
# 
# eval_clas_bnr('logistic regression',hra.lr.pdt,testing$real_left)

# random forest
# hra.rf.var <- colnames(training)
# hra.rf.f <- as.formula(paste("factor(left)~", paste(hra.rf.var[!hra.rf.var %in% "left"], collapse = " + ")))
# hra.rf <- randomForest(hra.rf.f,data=training)
# 
# hra.rf.pdt <- predict(hra.rf,testing)
# 
# eval_clas_bnr('random forest',hra.rf.pdt,testing$real_left)

# k means clustering
# hra.km <- kmeans(training,2)

# neural network
# hra.nn.var <- colnames(training_dm_sc)
# hra.nn.f <- as.formula(paste("left~", paste(hra.nn.var[!hra.nn.var %in% "left"], collapse = " + ")))
# hra.nn <- neuralnet(hra.nn.f,data=training_dm_sc,hidden=c(2,1),err.fct="ce",linear.output=FALSE,stepmax = 1000000)
# 
# testing_nn <- testing_dm_sc[,colnames(testing_dm)!="real_left"]
# hra.nn.pdt <- compute(hra.nn,testing_nn)
# hra.nn.pdt <- ifelse(hra.nn.pdt$net.result>0.5,1,0)
# 
# eval_clas_bnr('neural network',hra.nn.pdt,testing_dm$real_left)

# support vector machine
# hra.sv <- svm(left~.,data=training,type="C-classification")
# 
# hra.sv.pdt <- predict(hra.sv,testing)
# 
# eval_clas_bnr('support vector machine',hra.sv.pdt,testing$real_left)



# finished learning!
# get the time
end_time <- proc.time()
run_time <- end_time - start_time
print(paste("run time:",format(round(run_time["user.self"],digits = 10),nsmall=2),"sec"))

