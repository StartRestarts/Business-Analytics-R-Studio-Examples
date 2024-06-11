
## Please install "randomForest" package: install.packages("randomForest")
###################################################################################

install.packages("randomForest")

## load the library
library(randomForest)

## DATA EXPLORATION AND CLEANING
## load the wine data in R
## Be sure the wine dataset sits in your working directory
## explore the data set
wine <- WineDataset

dim(wine)
str(wine)
summary(wine)

wine$taste <- replace(wine$taste, wine$taste == 'bad', 0)
wine$taste <- replace(wine$taste, wine$taste == 'good', 1)
wine$taste <- replace(wine$taste, wine$taste == 'normal', 2)
wine$taste <- as.factor(wine$taste)

#check for missing data - Returns a binary split - Number of False vs. Number of True
#No of TRUE values represent missing data 
table(is.na(wine))

## Prepare the training vs. test split
## randomly choose 70% of the data set as training data
set.seed(27)
train.index <- sample(1:nrow(wine), 0.7*nrow(wine))
wine.train <- wine[train.index,]
dim(wine.train)

## select the other 30% as the testing data
wine.test <- wine[-train.index,]
dim(wine.test)

# Check for proportion of labels in both training and test split
prop.table(table(wine.train$taste))
prop.table(table(wine.test$taste))

## Fit decision model to training set
wine.rf.model <- randomForest(taste ~ ., data=wine.train, importance=TRUE, ntree=100, mtry=2)
print(wine.rf.model)

## show variable importance
importance(wine.rf.model)
varImpPlot(wine.rf.model)

## MODEL EVALUATION
## Predict test set outcomes, reporting class labels
wine.rf.predictions <- predict(wine.rf.model, wine.test, type="class")
## calculate the confusion matrix
wine.rf.confusion <- table(wine.rf.predictions, wine.test$taste)
print(wine.rf.confusion)
## accuracy
wine.rf.accuracy <- sum(diag(wine.rf.confusion)) / sum(wine.rf.confusion)
print(wine.rf.accuracy)

## EXERCISE
## Random forest has built-in feature selection.
## varImpPlot() function helps us visualize the importance of the features passed to 
## the model. Look at the importance table/graph, remove the least important predictor
## and re-build this model. Does it have similar performance? 
## If so, iterate, removing the new least important feature and re-building the model,
## until your model has much worse performance than the original one. 
## Imagine you are dealing with a much larger dataset, so memory and calculation time
## are something that you must be concerned about. In such a situation, what are the
## features you would choose to use in production?

## Please install "caret" package: install.packages("caret")
###################################################################################
# Using caret to create an even split and to perfrom model evaluation through cross validation
install.packages("caret")
library(caret)

# There are three steps for the model building using the caret package

#Step 1 - Create Data partition with stratified sampling

set.seed(1232)
indexes <- createDataPartition(wine$taste,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
wine.train <- wine[indexes,]
wine.test <- wine[-indexes,]



# Check for proportion of labels in both training and test split
prop.table(table(wine$taste))
prop.table(table(wine.train$taste))
prop.table(table(wine.test$taste))


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
caret.cv <- train(taste ~ ., 
                  data = wine.train,
                  method = "rf",
                  tuneGrid = tune.grid,
                  trControl = train.control)

stopCluster(cl)

# insert serial backend, otherwise error in repetetive tasks
registerDoSEQ()

# Make predictions on the test set using a randomForest model 
# trained on all rows of the training set using the 
# found optimal hyperparameter values.
preds.rf.caret <- predict(caret.cv, wine.test)

table(preds.rf.caret,wine.test$taste)
# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
Confusion.rf.caret<-confusionMatrix(preds.rf.caret, wine.test$taste)
Confusion.rf.caret
#To get the confusion matrix from the above output use the as.table()function

confusion.matrix<-as.table(Confusion.rf.caret)
confusion.matrix


