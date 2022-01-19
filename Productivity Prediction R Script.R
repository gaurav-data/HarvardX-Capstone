if(!require(tidyverse,caret,data.table)) install.packages("tidyverse","caret","data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2,dslabs,ggrepel)) install.packages("ggplot2","dslabs","ggrepel", repos = "http://cran.us.r-project.org")
if(!require(dplyr,lubridate,HistData)) install.packages("dplyr","lubridate","HistData", repos = "http://cran.us.r-project.org")
if(!require(purrr,pdftools,matrixStats)) install.packages("purrr","pdftools","matrixStats", repos = "http://cran.us.r-project.org")
if(!require(genefilter,randomForest,readxl)) install.packages("genefilter","randomForest","readxl", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(dslabs)
library(ggrepel)
library(dplyr)
library(lubridate)
library(HistData)
library(purrr)
library(pdftools)
library(matrixStats)
library(genefilter)
library(randomForest)
library(readxl)


#Code: Downloading Files from the Internet
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00597/garments_worker_productivity.csv"
tmp_filename <- tempfile()
download.file(url, tmp_filename)
dat <- read_csv(tmp_filename)

#Data pre exploration
str(dat)
head(dat)
dim(dat)

# Data Cleaning

dat<-dat%>% mutate(quarter=as.factor(quarter),
                   department=as.factor(department),day=as.factor(day),
                   team=as.factor(team))

##identifying values with NA
i<-1:15
col_NA<-sapply(i,function(l)
  dat%>% filter(is.na(dat[,l]))%>% summarise(n=n())
)
plot(i,col_NA)
names(dat[col_NA>0])

##replacing NA with 0
dat<-dat%>%mutate(wip=ifelse(is.na(wip),0,wip))
head(dat)

#Exploratory data Analysis

##summary statistics of the data:- 
summary(dat)

## distinct numbers of various parameters
dat%>%summarise(n_date=n_distinct(date),n_quarter=n_distinct(quarter),
                n_department=n_distinct(department),n_day=n_distinct(day),
                n_team=n_distinct(team))

## Trends of actual productivity w.r.t. various parameters
###By date
dat%>%group_by(department)%>%group_by(date)%>%group_by(team)%>%
  mutate(actual_productivity=mean(actual_productivity))%>%ggplot()+
  geom_point(aes(date,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))

###By Quarter
dat%>%group_by(quarter)%>%group_by(department)%>%group_by(team)%>%
  mutate(actual_productivity=mean(actual_productivity))%>%ggplot()+
  geom_point(aes(quarter,actual_productivity,color=team))+
  facet_grid(.~department)+ theme(axis.text.x=element_text(angle = 90,hjust=1))


###By Day
dat%>%group_by(department)%>%group_by(day)%>%group_by(team)%>%
  mutate(actual_productivity=mean(actual_productivity))%>%ggplot()+
  geom_point(aes(day,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))

### Above trends of actual productivity w.r.t. date, quarter and day is constant team wise

###targeted_productivity trend : shows increasing trend
dat%>% group_by(team)%>%mutate(actual_productivity=mean(actual_productivity), targeted_productivity=
                mean(targeted_productivity))%>%ggplot()+
  geom_point(aes(targeted_productivity,actual_productivity,color=team))+geom_abline()+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))

###smv: no trend

dat%>% ggplot()+
  geom_point(aes(smv,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))

###wip: shows no trends

dat%>% ggplot()+  geom_boxplot(aes(wip,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))


###over_time: no trends

dat%>% ggplot()+
  geom_boxplot(aes(over_time,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))


###incentive: no trends

dat%>% ggplot()+
  geom_boxplot(aes(incentive,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))



###idle_men: No trend

dat%>% ggplot()+
  geom_point(aes(idle_men,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))


###no_of_style_change:decreasing trends on sweing

dat%>% ggplot()+
  geom_point(aes(no_of_style_change,actual_productivity,color=team))+
  facet_grid(department~team)+theme(axis.text.x=element_text(angle = 90,hjust=1))


###no_of_workers: : no trend

dat%>% ggplot()+
  geom_point(aes(no_of_workers,actual_productivity,color=team))+
  facet_grid(.~department)+theme(axis.text.x=element_text(angle = 90,hjust=1))

#creating classification

## Classifying Productivity in classes 1-4. 4 means actual productivity 0.8, 3 means productivity is between 0.6 and 0.8, 2 means productivitu is between 0.4 and 0.6 and 1 means productivity is less than 0.4
## removing date and actual productivity for model building

rev_dat<-dat%>%mutate(rev_act_prod=ifelse(actual_productivity>0.8,4,
                                    ifelse(actual_productivity>0.6,3,
                                    ifelse(actual_productivity>0.4,2,1))))%>%
  select(-actual_productivity,-date)


head(rev_dat)

# Creating training and test datasets
# set.seed(1) if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(y=rev_dat$rev_act_prod, times = 1, p = 0.2, list = FALSE)
test_set <- rev_dat[test_index,]
train_set <- rev_dat[-test_index,]
head(test_set)
head(train_set)

# Defining RMSE for model testing
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# lm model: Linear regression model

fit_lm <- train(rev_act_prod ~ ., method="lm",data = train_set)
summary(fit_lm)
lm_preds <- predict(fit_lm, newdata = test_set)
rmse_lm <- RMSE(lm_preds, test_set$rev_act_prod)
rmse_results<-data_frame(method="lm model",RMSE=rmse_lm)
rmse_results %>% knitr::kable()

#K-nearest neighbors model
# set.seed(7)
set.seed(7, sample.kind = "Rounding") # simulate R 3.5
tuning <- data.frame(k = seq(3, 21, 2))
fit_knn <- train(rev_act_prod ~ .,data=train_set,
                 method = "knn", 
                 tuneGrid = tuning)
ggplot(fit_knn)
fit_knn$results
fit_knn$bestTune
summary(fit_knn)

#What is the accuracy of the kNN model on the test set?
knn_preds <- predict(fit_knn, test_set)
rmse_knn <- RMSE(knn_preds, test_set$rev_act_prod)
rmse_results<-bind_rows(rmse_results,data_frame(method="knn_model",RMSE=rmse_knn))
rmse_results %>% knitr::kable()


#Random forest model
# set.seed(9)
set.seed(9, sample.kind = "Rounding") # simulate R 3.5
tuning <- data.frame(mtry = c(5,7,9,11))
fit_rf <- train(rev_act_prod ~ .,data=train_set,
                method = "rf",
                tuneGrid = tuning,
                importance = TRUE)
plot(fit_rf)
fit_rf$bestTune
plot(fit_rf$finalModel)


#What is the accuracy of the random forest model on the test set?
rf_preds <- predict(fit_rf, test_set)
rmse_rf <- RMSE(rf_preds, test_set$rev_act_prod)
rmse_results<-bind_rows(rmse_results,data_frame(method="rf_model",RMSE=rmse_rf))
rmse_results %>% knitr::kable()

#What is the most important variable in the random forest model?
varImp(fit_rf)

