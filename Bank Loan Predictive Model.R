#Clear Environment memory before beginning analysis
rm(list=ls())
# Install necessary packages

install.packages("caret")
library(caret)
library(tidyverse)
# Set seed so that results of this session are reproducible
set.seed(5)
#Load data:
data<-read_csv("UniversalBank.csv")

# Check out data attributes 
head(data)
str(data)
summary(data)

# ID and ZIP code variables will not help with determining whether or not loan will be accepted
# We'll need to drop ID and ZIP code columns:
data <-subset(data, select=-c(ID, ZIP.Code))
#I'll need to rename columns before making any changes to them:
names(data)[5]<-"ZIP"
names(data)[10]<-"Loan"
names(data)[11]<-"SA"
names(data)[12]<-"CD"
#drop ID and ZIP Code columns:
data <-subset(data, select=-c(ID, ZIP))
head(data)
# Education, Our response variable, Loan, and Account type variables are categorical variables, I will declare them as such
data$Education<-as.factor(data$Education)
data$SA<-as.factor(data$SA)
data$Loan<-as.factor(data$Loan)
data$CD<-as.factor(data$CD)
data$Online<-as.factor(data$Online)
data$CreditCard<-as.factor(data$CreditCard)

# Data can now be split into training and test data sets
# I will use and 80% training, 20% testing split on this dataset
ind<-createDataPartition(data$Loan,p=0.8,list=F)
train<-data[ind,]
test<-data[-ind,]

# Scale and standardize data so relationship between response and explanator variables is like-for-like
scaled_df<-preProcess(train[,setdiff(names(train), "Loan")],method=c("center","scale"))
train<-predict(scaled_df, train)
test<-predict(scaled_df, test)

#Convert all categorical variables into dummy to allow for testing model accuracy using predict function
dummies<- dummyVars(Loan~., data=data)

#declare test and training variables, plus response variables
x.train=predict(dummies,newdata=train)
y.train=train$Loan
x.test=predict(dummies, newdata=test)
y.test=test$Loan

#Load SVM package
install.packages("e1071")
library(e1071)

model <- svm(x=x.train, y=y.train, type="C-classification", kernel="linear", cost=10)
summary(model)
attributes(model)

#For our SVM linear classifier, out of 5000 observations, 437 are support vectors - on the classfication hyperplane at risk of mis-classification
# dropping cost to 1 reduces SVMs by increasing margin between classifications
# Trade-off of increasing increasing margin is increased likelihood of misclassfication

pred_train<-predict(model,x.train)
pred_test<-predict(model, x.test)

confusionMatrix(pred_train, y.train)
#Our training dataset confusion matrix gives an accuracy measure of 96.3% for our linear classification model with a cost of 10:
# Out of 4000 observations:
# We have 255 true positives, 129 false negatives, 19 false positives and 3597 true negatives
# This shows that our model is highly cautious - it reduces the likelihood of suggesting someone that would
# Why does model not predict that anyone will get a loan?
# Our confidence intervals also suggest linear model is a good fit for our dataset

confusionMatrix(pred_test, y.test)

#Our test dataset confusion matrix also gives an accuracy measure of 96.6% for our test set
#Out of 1000 observations:
# We have 66 true positives, 30 false negatives, 4 false positives and 900 true negatives
# This shows that our model is highly cautious - it reduces the likelihood of suggesting someone that would
# Why does model not predict that anyone will get a loan?
# Our confidence intervals also suggest linear model is a good fit for our dataset

#Trying a different model:

model2<-svm(x=x.train, y=y.train, "C-classification", kernel="radial", cost=10, gamma=0.1)
summary(model2)
model3<-ksvm(x.train,y.train, type="C-svc", kernel="rbfdot", kpar="automatic", C=10, cross=5)
summary(model3)
model3
pred_train2<-predict(model3,x.train)
pred_test2<-predict(model3,x.test)

confusionMatrix(pred_train2, y.train)
#Our training dataset confusion matrix gives an accuracy measure of 99.12% for our linear classification model with a cost of 10:
# Out of 4000 observations:
# We have 353 true positives, 31 false negatives, 4 false positives and 3612 true negatives

confusionMatrix(pred_test2, y.test)
#Out of 1000 observations:
# We have 88 true positives, 8 false negatives, 9 false positives and 895 true negatives

tuneResult <- tune(svm, train.x=x.train,train.y=y.train, ranges=list(gamma = 10^(-3:-1), cost=2^(2:3)), class.weights=c("0"=1, "1"=10),tunecontrol=tune.control(cross=3))
print(tuneResult)
summary(tuneResult)

tuned_model<-tuneResult$best.model;tuned_model
#Our "best" model has 
pred_train3<-predict(tuned_model,x.train)
pred_test3<- predict(tuned_model,x.test)

confusionMatrix(pred_train3, y.train)
#Our training dataset confusion matrix gives an accuracy measure of 99.12% for our linear classification model with a cost of 10:
# Out of 4000 observations:
# We have 383 true positives, 1 false negative, 27 false positives and 3589 true negatives

confusionMatrix(pred_test3, y.test)
#Out of 1000 observations:
# We have 85 true positives, 11 false negatives, 23 false positives and 881 true negatives

#Considering the decrease in true positives, and increase in false positives in test set, is this model necessarily the bets one to use in reality?

