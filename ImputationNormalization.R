

install.packages("imputeTS")
#(Q1) call the installed package from your library to proceed

#(Q2) Read the countries data file and assign the data frame to a R object 'countries'
#(Q3) Find the summary of the data frame using 'summary' function

###################################################################################################
countries$Literacy_impute<-na_mean(countries$Literacy, option="mean")

#(Q4) impute the missing values in 'climate' variable by taking the median of the column


#(Q5) Similarly impute the other columns with mean


###################################################################################################
################################################
#Data transformation - min-max normalization and z-score normalization

#Min-max normalization for the literacy impute column
min<-min(countries$Literacy_impute)
max<-max(countries$Literacy_impute)
Value<-countries$Literacy_impute

norm_value<- (Value-min)/(max-min)

countries$Literacy_norm<-norm_value

################################################

#Z-score normalization of imputed literacy column using the scale function
#read ?scale() for more information

zscore_norm<-scale(countries$Literacy_impute)
countries$Literacy_zscore_norm<-zscore_norm

#(Q6) Try for the other columns for practice

###############################################
#(Q7) Convert the entire dataset into a Zscore scaled dataset

