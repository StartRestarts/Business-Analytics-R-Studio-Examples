install.packages("ggplot2")
library(ggplot2)

# Set up factors.
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$Survived <- as.factor(titanic$Survived)
titanic$Sex <- as.factor(titanic$Sex)
titanic$Embarked <- as.factor(titanic$Embarked)


bank$job <- as.factor(bank$job)
bank$marital <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$y <- as.factor(bank$y)

ggplot(bank, aes(x = job, fill = y)) +
  geom_bar()
ggplot(bank, aes(x = marital, fill = y)) +
  geom_bar()
ggplot(bank, aes(x = education, fill = y)) +
  geom_bar()
ggplot(bank, aes(x = age, fill = y)) +
  geom_histogram(binwidth = 5)

#
# We'll start our visual analysis of the data focusing on questions
# related to survival rates. Specifically, these questions will use
# the factor (i.e., categorical) variables in the data. Factor data
# is very common in the business context and ggplot2 offers many
# powerful features for visualizing factor data.
#


#
# First question - What was the survival rate? 
#
# As Survived is a factor (i.e., categorical) variable, a bar chart 
# is a great visualization to use.
#

ggplot(titanic, aes(x = Survived)) + 
  geom_bar()

# Add some customization for labels and theme.
ggplot(titanic, aes(x = Survived)) + 
  geom_bar() +
  labs(y = "Passenger Count",
       title = "Titanic Survival Rates")


#
# Second question - What was the survival rate by gender? 
#
# We can use color to look at two aspects (i.e., dimensions)
# of the data simultaneously.
#
ggplot(titanic, aes(x = Sex, fill = Survived)) + 
  geom_bar() +
  labs(y = "Passenger Count",
       title = "Titanic Survival Rates by Sex")


#
# Third question - What was the survival rate by class of ticket? 
#
ggplot(titanic, aes(x = Pclass, fill = Survived)) + 
  geom_bar() +
  labs(y = "Passenger Count",
       title = "Titanic Survival Rates by Pclass")


#
# Fourth question - What was the survival rate by class of ticket
# and gender?
#
# We can leverage facets to further segment the data and enable
# "visual drill-down" into the data.
#
ggplot(titanic, aes(x = Sex, fill = Survived)) + 
  facet_wrap(~ Pclass) +
  geom_bar() +
  labs(y = "Passenger Count",
       title = "Titanic Survival Rates by Pclass and Sex")




#
# Next, we'll move on to visualizing continuous (i.e., numeric)
# data using ggplot2. We'll explore visualizations of single 
# numeric variables (i.e., columns) and also illustrate how
# ggplot2 enables visual drill-down on numeric data.
#


#
# Fifth Question - What is the distribution of passenger ages?
#
# The histogram is a staple of visualizing numeric data as it very 
# powerfully communicates the distrubtion of a variable (i.e., column).
#
ggplot(titanic, aes(x = Age)) +
  geom_histogram(binwidth = 5) +
  labs(y = "Passenger Count",
       x = "Age (binwidth = 5)",
       title = "Titanic Age Distribtion")


#
# Sixth Question - What are the survival rates by age?
#
ggplot(titanic, aes(x = Age, fill = Survived)) +
  geom_histogram(binwidth = 5) +
  labs(y = "Passenger Count",
       x = "Age (binwidth = 5)",
       title = "Titanic Survival Rates by Age")



# Seventh Question - What is the survival rates by age when segmented
# by gender and class of ticket?
#
# A related visualization to the histogram is a density plot. Think of
# a density plot as a smoothed version of the histogram. Using ggplot2
# we can use facets to allow for visual drill-down via density plots.
#
ggplot(titanic, aes(x = Age, fill = Survived)) +
 facet_wrap(Pclass~Sex) +
  geom_density(alpha=0.5) +
  labs(y = "Age",
       x = "Survived",
       title = "Titanic Survival Rates by Age, Pclass and Sex")

# If you prefer histograms, no problem!
ggplot(titanic, aes(x = Age, fill = Survived)) +
  facet_wrap(Sex ~ Pclass) +
  geom_histogram(binwidth = 5) +
  labs(y = "Age",
       x = "Survived",
       title = "Titanic Survival Rates by Age, Pclass and Sex")

#Excercise: Using the mtcars dataset create a scatter plot that plots mpg and hp on the mtcars dataset and segment it using cycl and gears
#What are your insights


