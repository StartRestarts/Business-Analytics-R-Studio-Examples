
## Please install "caret" package: install.packages("caret")
## install.packages("randomForest")
## Using caret to create an even split and to perfrom model building

## load the library
install.packages("randomForest")
library(randomForest)

## DATA EXPLORATION AND CLEANING
## load the wine data in R
## Be sure the winde dataset sits in your working directory
wine <- WineDataset
wine$taste <- as.factor(wine$taste)
 ## explore the data set
dim(wine)
str(wine)
summary(wine)

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
prop.table(table(wine$taste))
prop.table(table(wine.train$taste))
prop.table(table(wine.test$taste))

wine.train$taste <- as.factor(wine.train$taste)

## Fit decision model to training set
wine.rf.model <- randomForest(taste ~ ., data=wine.train, importance=TRUE)
print(wine.rf.model)

## show variable importance
importance(wine.rf.model)
varImpPlot(wine.rf.model)

predictions <- predict(wine.rf.model, data=wine.test)
print(predictions)

#comparsion table
wine.comparison <- wine.test
wine.comparison$Prediction <- predictions
wine.comparison[ , c("Taste", "Predictions")]

install.packages("caret")
library(caret)
set.seed(1232)

indexes <- createDataPartition(wine$taste,times = 1,p = 0.7,list = FALSE)
wine.train <- wine[indexes,]
wine.test <- wine[-indexes,]

# Check for proportion of labels in both training and test split
prop.table(table(wine$taste))
prop.table(table(wine.train$taste))
prop.table(table(wine.test$taste))

caret.cv <- train(taste ~ ., data = wine.train, method = "rf")


# Alternatively, training rf using Gaussian method
#caret.cv.b <- train(taste ~ ., data = wine.train, method = "gbm") 



