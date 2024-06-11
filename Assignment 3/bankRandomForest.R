#Load the Library
install.packages("randomForest")
install.packages("caret")
install.packages("dplyr")
library(dplyr)
library(randomForest)
library(caret)


#Data Exploration
bank <- bank_clean
bank$y<- as.factor(bank$y)

dim(bank)
str(bank)
summary(bank)

#check for missing data
table(is.na(bank))

## Please install "caret" package: install.packages("caret")
###################################################################################
# Using caret to create an even split and to perform model evaluation through cross validation

# There are three steps for the model building using the caret package

#Step 1 - Create Data partition with stratified sampling

set.seed(420)
indexes <- createDataPartition(bank$y,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
bank.train <- bank[indexes,]
bank.test <- bank[-indexes,]



# Check for proportion of labels in both training and test split
prop.table(table(bank$y))
prop.table(table(bank.train$y))
prop.table(table(bank.test$y))

bank.rf.model <- randomForest(y ~ ., data=bank.train, importance=TRUE, ntree=100, mtry=2)
print(bank.rf.model)

importance(bank.rf.model)
varImpPlot(bank.rf.model)

#=================================================================
# Train Model -Random Forest
#=================================================================

# Set up caret to perform 10-fold cross validation repeated 3 
# times and to use a grid search for optimal model hyperparamter
# values.

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid",
                              allowParallel = T)

# Leverage a grid search of hyperparameters for randomForest. See 
# the following presentation for more information:

tune.grid <- expand.grid(mtry = c(3:6))

View(tune.grid)

#Let's run it parallely
install.packages("doParallel")

library(doParallel)

cl <- makePSOCKcluster(8)
registerDoParallel(cl)

# Train the randomforest model using 10-fold CV repeated 3 times 
# and a hyperparameter grid search to train the optimal model.
caret.cv <- train(y ~ ., 
                  data = bank.train,
                  method = "rf",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)

# insert serial backend, otherwise error in repetetive tasks
registerDoSEQ()

# Make predictions on the test set using a randomForest model 
# trained on all rows of the training set using the 
# found optimal hyperparameter values.
preds.rf.caret <- predict(caret.cv, bank.test)

table(preds.rf.caret,bank.test$y)
# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
Confusion.rf.caret<-confusionMatrix(preds.rf.caret, bank.test$y)
Confusion.rf.caret
#To get the confusion matrix from the above output use the as.table()function

confusion.matrix<-as.table(Confusion.rf.caret)
confusion.matrix

##Calculate accuracy, precision, recall, F1
#Accuracy
bank.accuracy <- sum(diag(confusion.matrix)) / sum(confusion.matrix)
print(bank.accuracy)

#Precision per class
bank.precision.A <- confusion.matrix[1,1] / sum(confusion.matrix[,1])
print(bank.precision.A)

bank.precision.B <- confusion.matrix[2,2] / sum(confusion.matrix[,2])
print(bank.precision.B)

#Overall precision
overall.precision<-(bank.precision.A+bank.precision.B)/2
print(overall.precision)

#Recall per class
bank.recall.A <- confusion.matrix[1,1] / sum(confusion.matrix[1,])
print(bank.recall.A)

bank.recall.B <- confusion.matrix[2,2] / sum(confusion.matrix[2,])
print(bank.recall.B)

#Overall recall
overall.recall<-(bank.recall.A+bank.recall.B)/2
print(overall.recall)

bank.f1 <- 2 * overall.precision * overall.recall / (overall.precision + overall.recall)
print(bank.f1)

