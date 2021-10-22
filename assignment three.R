install.packages('rminer') #For holdout method
install.packages("ROSE") # for performing undersampling
install.packages('randomForest')
install.packages("MLmetrics")
library(ROSE)
library(rminer)
library(dplyr)
library(caret)
library(tidyverse)
library(randomForest)
library(rpart)
library(rpart.plot) 
library(corrplot)
library(cluster)
library(MLmetrics)


framingham <- read.delim("framingham.csv",sep = ",")
Data <- select(framingham, age, male)


# standarizing age feild
Data$age <- scale(Data$age)

#creating k-means clusters
set.seed(555)

Cluster_kmean <- kmeans(Data, 4, nstart = 20)

#plotting the clusters
Data$clusters <-Cluster_kmean$cluster 
Data$clusters <- factor(Data$clusters)

 
Cluster_kmean$cluster <- factor(Cluster_kmean$cluster)
ggplot(Data, aes(male,age, color = clusters)) + 
  geom_point(alpha = 0.4, size = 3.5) + geom_point(col = Cluster_kmean$cluster) + 
  scale_color_manual(values = c('black', 'red', 'green','yellow'))

#plotting elbow curve
set.seed(50)

wss <- (nrow(Data)-1)*sum(apply(Data[,1:2],2,var))
for (i in 2:15) {
  wss[i] <- sum(kmeans(Data[,1:2],centers=i)$withinss)
}
plot(1:15, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")


# silouette score 
set.seed(50)
silhouette_score <- function(k){
  km <- kmeans(Data, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(Data))
  mean(ss[, 3])
}
k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)

Data1 <- mutate_all(Data, function(x) as.numeric(as.character(x)))

silhouette_coeff <- silhouette(as.numeric(Cluster_kmean$cluster),dist(Data1))

mean(silhouette_coeff[, 3])
Cluster_kmean$tot.withinss

#==============================================================================
# Clustering with k=6
set.seed(50)

Cluster_kmean <- kmeans(Data, 6, nstart = 20)

#plotting the clusters
Data$clusters <-Cluster_kmean$cluster 
Data$clusters <- factor(Data$clusters)


Cluster_kmean$cluster <- factor(Cluster_kmean$cluster)
ggplot(Data, aes(male,age, color = clusters)) + 
  geom_point(alpha = 0.4, size = 3.5) + geom_point(col = Cluster_kmean$cluster) + 
  scale_color_manual(values = c('black', 'red', 'green','yellow','blue','orange'))


silhouette_coeff <- silhouette(as.numeric(Cluster_kmean$cluster),dist(Data1))

mean(silhouette_coeff[, 3])
Cluster_kmean$tot.withinss
	


#========================================================================================================
#Part B

Churn_Data <- read.delim("customer_churn.csv",sep = ",")

Churn_Data<-na.omit(Churn_Data)

H <- holdout(Churn_Data$Churn, ratio = 0.67, internalsplit = FALSE, mode = "stratified", iter = 1, 
                         seed = 555, window=10, increment=1) 

A<- as.vector(table(Churn_Data[H$tr,]$Churn))

B<-as.vector(table(Churn_Data[H$ts,]$Churn))



#plotting the bar graph 
# Create the input vectors.
colors = c("green","red")
Set <- c("Training Set","Testing Set")
Churn <- c("No","Yes")

# Create the matrix of the values.
Values <- matrix(c(A,B), nrow = 2, ncol = 2, byrow = TRUE)



# Create the bar chart
barplot(Values, main = "total Count", names.arg = Set, xlab = "Set", ylab = "Count", col = colors)

# Add the legend to the chart
legend("topright", Churn, cex = 1.3, fill = colors)


Training_set<-as.data.frame(Churn_Data[H$tr,])
Testing_set<-Churn_Data[H$ts,]

#making the percent of the true class 0.3 of the whole data 

data.balanced.ou <- ovun.sample(Churn~., data=Training_set,
                                p=0.3 ,
                                 seed=1, method="both")$data

#selecting revant predictors

df <-subset(data.balanced.ou, select = -c(customerID,gender,PaymentMethod,PaperlessBilling))
df_test<-subset(Testing_set, select = -c(customerID,gender,PaymentMethod,PaperlessBilling))

df_test$Churn<-factor(df_test$Churn, levels = c("No","Yes"), 
                                labels = c("No","Yes"))




        
DF<- df
DF$Churn<- factor(DF$Churn)










library(caret)
ctrl <- trainControl(method = "repeatedcv",
                     number = 10, repeats = 10,
                     selectionFunction = "best",
                     savePredictions = TRUE,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# auto-tune a random forest
grid_rf <- expand.grid(mtry = c(2, 4, 8, 16))


RNGversion("3.5.2") 
set.seed(300)
m_rf <- train(Churn ~ ., data = DF, method = "rf",
              metric = "ROC", trControl = ctrl,
              tuneGrid = grid_rf)
m_rf

# auto-tune a boosted C5.0 decision tree
grid_c50 <- expand.grid(model = "tree",
                        trials = c(10, 25, 50, 100),
                        winnow = FALSE)

RNGversion("3.5.2") # use an older random number generator to match the book
set.seed(300)
m_c50 <- train(Churn ~ ., data = DF, method = "C5.0",
               metric = "ROC", trControl = ctrl,
               tuneGrid = grid_c50)
m_c50

# compare their ROC curves
library(pROC)
roc_rf <- roc(m_rf$pred$obs, m_rf$pred$Yes)
roc_c50 <- roc(m_c50$pred$obs, m_c50$pred$Yes)

plot(roc_rf, col = "red", legacy.axes = TRUE)
plot(roc_c50, col = "blue", add = TRUE)
legend("topleft", c("Decision tree","Random Foreset"), cex = .7, fill = c("blue",'red'))



# Confusion matrix for decision tree
test_pred_gini <- predict(m_c50, newdata = df_test)
confusionMatrix(test_pred_gini, df_test$Churn)

# Confusion Matrix for Random forest
test_pred_rfover=predict(m_rf, newdata=df_test )
confusionMatrix(test_pred_rfover, df_test$Churn)
#F1 Score for decision tree
F1_Score(df_test$Churn,test_pred_gini, positive = "Yes")

# F1 score for random forest 
F1_Score(df_test$Churn,test_pred_rfover, positive = "Yes")
