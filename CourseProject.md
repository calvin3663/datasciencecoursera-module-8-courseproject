# Practical Machine Learning: Course Project
Calvin Chin  

# Synopsis 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. In this project, the author will use data from accelerometers to build a machine learning algorithm that will predict activity quality from activity monitors.

# Data

The **training** data for this project are available here: 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The **test** data are available here: 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

NOTE: The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>

# Preparing the environment


```r
# Load the required libraries
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
trainingRAW <- read.csv(file="pml-training.csv")
testingRAW <- read.csv(file="pml-testing.csv")
```


# Pre-processing


```r
# Remove first 7 columns that are not relevant to the model building
trainingNew <- trainingRAW[ , -c(1:7)]


# Remove columns that have too many NAs
isNA <- sapply(trainingNew, function(x) mean(is.na(x))) > 0.95

trainingNew <- trainingNew[ , isNA==FALSE]


# Remove columns with Near Zero Variance
indexNZV <- nearZeroVar(trainingNew)

trainingNew <- trainingNew[ , -indexNZV]
```


# Model Building


```r
# Create train and test data partitions
inTrain <- createDataPartition(y=trainingNew$classe, p=0.7, list=FALSE)

training <- trainingNew[inTrain,]
testing <- trainingNew[-inTrain,]


# Create Model Fit using Random Forest. Cross validation (i.e cv) is used as trainControl method, with number of resampling iterations set to 3 for faster performance
modFit <- train(classe ~ ., data=training, method="rf", trControl=trainControl(method="cv", number=3))

print(modFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.8%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3900    4    1    0    1 0.001536098
## B   20 2628    7    3    0 0.011286682
## C    0   13 2376    7    0 0.008347245
## D    0    2   32 2217    1 0.015541741
## E    0    2    5   12 2506 0.007524752
```


# Cross Validation

```r
pred_testing <- predict(modFit, newdata=testing)

confusionMatrix(pred_testing, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    9    0    0    0
##          B    3 1129    4    0    0
##          C    0    1 1021    9    1
##          D    0    0    1  955    9
##          E    0    0    0    0 1072
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9912   0.9951   0.9907   0.9908
## Specificity            0.9979   0.9985   0.9977   0.9980   1.0000
## Pos Pred Value         0.9946   0.9938   0.9893   0.9896   1.0000
## Neg Pred Value         0.9993   0.9979   0.9990   0.9982   0.9979
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1918   0.1735   0.1623   0.1822
## Detection Prevalence   0.2855   0.1930   0.1754   0.1640   0.1822
## Balanced Accuracy      0.9980   0.9949   0.9964   0.9943   0.9954
```
Cross Validation results show overall accuracy of the model is >99%


# Now, apply Model Fit to predict the 'classe' value for the test cases provided

```r
# For the Testing dataset, Extract only the same columns prepared from the Training dataset
requiredColumns <- names(trainingNew[,1:length(trainingNew)-1])
testingNew <- testingRAW[ , requiredColumns]


pred_testingNew <- predict(modFit, newdata=testingNew)

# First print prediction results onto the screen 
print(pred_testingNew)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
# Next, write prediction results onto text files required for Course Project Part 2 Submission (1 charcter in each file)

for(i in 1:length(pred_testingNew))
{
    filename = paste0("problem_id_", i, ".txt")
    write.table(pred_testingNew[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
```


