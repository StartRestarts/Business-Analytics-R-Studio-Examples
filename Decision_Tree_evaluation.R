
###################################################################################

## Objective:  iris species classification with decision tree
## Data source: iris data set (included in R)
## Please install "rpart" package: install.packages("rpart")

###################################################################################

## load the library
library(rpart)

## DATA EXPLORATION
## load the iris data in R
data(iris)
## explore the data set
str(iris)
dim(iris)
summary(iris)
################################################################################
## BUILD MODEL
## randomly choose 70% of the data set as training data
set.seed(777)#Why do you need to set seed?

train.index <- sample(1:nrow(iris), 0.7*nrow(iris))
iris.train <- iris[train.index,]
dim(iris.train)
## select the 30% left as the testing data
iris.test <- iris[-train.index,]
dim(iris.test)

# Default decision tree model
    # Builds a decision tree from the iris dataset to predict
    # species given all other columns as predictors
iris.tree <- rpart(Species~.,data=iris.train)

# Reports the model
print(iris.tree)

## VISUALIZE THE MODEL
## plot the tree structure
plot(iris.tree, margin=c(.1))
title(main = "Decision Tree Model of Iris Data")
text(iris.tree, use.n = TRUE)
## print the tree structure
summary(iris.tree)

## MODEL EVALUATION
## make prediction using decision model
iris.predictions <- predict(iris.tree, iris.test, type = "class")
head(iris.predictions)

## Comparison table
iris.comparison <- iris.test
iris.comparison$Predictions <- iris.predictions
iris.comparison[ , c("Species", "Predictions")]

## View misclassified rows
disagreement.index <- iris.comparison$Species != iris.comparison$Predictions
iris.comparison[disagreement.index,]

## If instead you wanted probabilities.
# iris.predictions <- predict(iris.tree, iris.test, type = "prob")

## Extract the test data species to build the confusion matrix
iris.confusion <- table(iris.predictions, iris.test$Species)
print(iris.confusion)

## calculate accuracy, precision, recall, F1
#Accuracy
iris.accuracy <- sum(diag(iris.confusion)) / sum(iris.confusion)
print(iris.accuracy)

#Precision per class
iris.precision.A <- iris.confusion[1,1] / sum(iris.confusion[,1])
print(iris.precision.A)

iris.precision.B <- iris.confusion[2,2] / sum(iris.confusion[,2])
print(iris.precision.B)

iris.precision.C <- iris.confusion[3,3] / sum(iris.confusion[,3])
print(iris.precision.C)

#Overall precision
overall.precision<-(iris.precision.A+iris.precision.B+iris.precision.C)/3
print(overall.precision)

#Recall per class
iris.recall.A <- iris.confusion[1,1] / sum(iris.confusion[1,])
print(iris.recall.A)

iris.recall.B <- iris.confusion[2,2] / sum(iris.confusion[2,])
print(iris.recall.B)

iris.recall.C <- iris.confusion[3,3] / sum(iris.confusion[3,])
print(iris.recall.C)

#Overall recall
overall.recall<-(iris.recall.A+iris.recall.B+iris.recall.C)/3
print(overall.recall)
  
iris.f1 <- 2 * overall.precision * overall.recall / (overall.precision + overall.recall)
print(iris.f1)



#### Parameter Tuning ####

## Setting control parameters for rpart
## Check ?rpart.control for what the parameters do
tree.params <- rpart.control(minsplit=20, minbucket=7, maxdepth=30, cp=0.01)

## Fit decision model to training set
## Use parameters from above and Gini index for splitting
iris.tree <- rpart(Species ~ ., data = iris.train, 
                       control=tree.params, parms=list(split="gini"))


