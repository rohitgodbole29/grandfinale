Predicting\_excercise
================
RG
February 5, 2018

R Markdown
----------

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:

``` r
library(caret)
library(rpart)
library(rattle)
library(dplyr)
library(here)
library(tibble)
library(tidyr)
library(ggplot2)
library(GGally)
library(reshape)
```

Load the training data.
-----------------------

we are also creating a partition called "val" so that we can have private data for ensembling different models. The validation data is a seperate partition within the training data. This is not the traditional nomenclature for validation data, however it may be better understood as staging data. IN any case this seperate partition allows us to keep the testing data , untouched till the final stage.

``` r
dat_train = read.csv(file= here("pml-training.csv"), header=TRUE)
inTrain<- createDataPartition(y = dat_train$classe,p = 0.9,list = FALSE)
dat_train<-dat_train[inTrain,]
dat_train<-na.omit(dat_train)
dat_val<-dat_train[-inTrain,]
dat_val<-na.omit(dat_val)
```

Load the test data.
-------------------

Here we have choosen only those predictors that have non-NA values in the test data set. We also got rid of obviously unrelated predictors like timestamp and username , which hopefully would have no bearing on the outcome. This is confirmed by the plot below. ![](index_files/figure-markdown_github/unnamed-chunk-1-1.png)

Subset the Training set and the validation data.
------------------------------------------------

Configure Parallel Processing
-----------------------------

``` r
library(parallel)
library(doParallel)
cluster<-makeCluster(detectCores()-1) # one core for the OS.
registerDoParallel(cluster)
```

Model1 :Build a Random Forest model & Confusion Matrix
------------------------------------------------------

``` r
fitControl<-trainControl(method = "cv",number = 5,allowParallel = TRUE)
modFit<-train(form = classe~.,data = dat_train3,method="rf",prox=TRUE,trControl=fitControl)
pred<-predict(modFit,dat_val3)
confusionMatrix(pred,dat_val3$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  A  B  C  D  E
    ##          A  6  0  0  0  0
    ##          B  0  6  0  0  0
    ##          C  0  0  6  0  0
    ##          D  0  0  0 12  0
    ##          E  0  0  0  0  6
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9026, 1)
    ##     No Information Rate : 0.3333     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.1667   0.1667   0.1667   0.3333   0.1667
    ## Detection Rate         0.1667   0.1667   0.1667   0.3333   0.1667
    ## Detection Prevalence   0.1667   0.1667   0.1667   0.3333   0.1667
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Out of sample Error : An estimate can be provided based on the accuracy in prediction on the Val dat set.
---------------------------------------------------------------------------------------------------------

We notice that the 95% CI is (0.8806 ,1). This means that the in the worst case we are 95% confident that the accuracy wont be lower than 88%. That gives us a mean accuracy of about 94% or an expected error of about 6%.

Model2 : Build a Naive Bayes Model
----------------------------------

``` r
modnb = train(form=classe~.,data = na.omit(dat_train3),method="nb")
```

Predict model2 : NB on Validation and generate confusion matrix
---------------------------------------------------------------

``` r
pnb= predict(modnb,dat_val3)
confusionMatrix(pnb,dat_val3$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction A B C D E
    ##          A 6 0 0 1 1
    ##          B 0 5 0 0 1
    ##          C 0 0 6 5 1
    ##          D 0 1 0 6 0
    ##          E 0 0 0 0 3
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7222         
    ##                  95% CI : (0.5481, 0.858)
    ##     No Information Rate : 0.3333         
    ##     P-Value [Acc > NIR] : 2.115e-06      
    ##                                          
    ##                   Kappa : 0.6532         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.8333   1.0000   0.5000  0.50000
    ## Specificity            0.9333   0.9667   0.8000   0.9583  1.00000
    ## Pos Pred Value         0.7500   0.8333   0.5000   0.8571  1.00000
    ## Neg Pred Value         1.0000   0.9667   1.0000   0.7931  0.90909
    ## Prevalence             0.1667   0.1667   0.1667   0.3333  0.16667
    ## Detection Rate         0.1667   0.1389   0.1667   0.1667  0.08333
    ## Detection Prevalence   0.2222   0.1667   0.3333   0.1944  0.08333
    ## Balanced Accuracy      0.9667   0.9000   0.9000   0.7292  0.75000

Model3 : Use a boosting algorithm.
----------------------------------

Predicting on Validation using boosting
---------------------------------------

``` r
pboost= predict(modboost,dat_val3)
confusionMatrix(pboost,dat_val3$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  A  B  C  D  E
    ##          A  6  0  0  0  0
    ##          B  0  6  0  0  0
    ##          C  0  0  6  0  0
    ##          D  0  0  0 12  0
    ##          E  0  0  0  0  6
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9026, 1)
    ##     No Information Rate : 0.3333     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.1667   0.1667   0.1667   0.3333   0.1667
    ## Detection Rate         0.1667   0.1667   0.1667   0.3333   0.1667
    ## Detection Prevalence   0.1667   0.1667   0.1667   0.3333   0.1667
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Model4 : Use a model stacking approach.
---------------------------------------

deregsiter from the parallel processing
---------------------------------------

``` r
stopCluster(cluster)
registerDoSEQ()
```

Predict on the Test Set using the Random Forest Algo.
-----------------------------------------------------

``` r
predTest<-predict(modFit,dat_test1)
predTest
```

    ##  [1] C A C A A E C B A A B C B A E E A D A B
    ## Levels: A B C D E

Predict using Naive Bayes on the Test data.
-------------------------------------------

``` r
pnbtest= predict(modnb,dat_test1)
pnbtest
```

    ##  [1] A C C C C E C C A A A C B A E B A B A B
    ## Levels: A B C D E

Predict on the test set using the boosting algo.
------------------------------------------------

``` r
pbootest= predict(modboost,dat_test1)
pbootest
```

    ##  [1] C A B A A E D B A A B C B A E E A B A B
    ## Levels: A B C D E

Predicting the combined model ( all 3 models combined) on the test data
-----------------------------------------------------------------------

``` r
# we shall use the values of the predictions of the earlier models on the test data.Accordingly.
predVDF<- data.frame(pred= predTest,pnb=pnbtest, pboost= pbootest)
pstackedfinal= predict(pcombined,predVDF)
pstackedfinal
```

    ##  [1] B A B A A B A B A A B B B A B B A B A B
    ## Levels: A B C D E

The Random Forest model seems to fit better than the combined model, since the classification is coarse.

This gives a good variation. Let us understand the agreement between the boosting and random forest approaches.

``` r
qplot(x = predTest,y = pbootest,data = dat_test1)
```

![](index_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
table(predTest,pbootest)
```

    ##         pbootest
    ## predTest A B C D E
    ##        A 8 0 0 0 0
    ##        B 0 4 0 0 0
    ##        C 0 1 2 1 0
    ##        D 0 1 0 0 0
    ##        E 0 0 0 0 3

``` r
confusionMatrix(predTest,pbootest)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction A B C D E
    ##          A 8 0 0 0 0
    ##          B 0 4 0 0 0
    ##          C 0 1 2 1 0
    ##          D 0 1 0 0 0
    ##          E 0 0 0 0 3
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.85            
    ##                  95% CI : (0.6211, 0.9679)
    ##     No Information Rate : 0.4             
    ##     P-Value [Acc > NIR] : 4.734e-05       
    ##                                           
    ##                   Kappa : 0.7959          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity               1.0   0.6667   1.0000   0.0000     1.00
    ## Specificity               1.0   1.0000   0.8889   0.9474     1.00
    ## Pos Pred Value            1.0   1.0000   0.5000   0.0000     1.00
    ## Neg Pred Value            1.0   0.8750   1.0000   0.9474     1.00
    ## Prevalence                0.4   0.3000   0.1000   0.0500     0.15
    ## Detection Rate            0.4   0.2000   0.1000   0.0000     0.15
    ## Detection Prevalence      0.4   0.2000   0.2000   0.0500     0.15
    ## Balanced Accuracy         1.0   0.8333   0.9444   0.4737     1.00

Let us try voting from the 4 approaches considered as above.
------------------------------------------------------------

-- predTest from Random Forest :In-sample--&gt;accuracy = 1 -- pnbtest from NB: In-Sample --&gt; accuracy = 0.5263 -- pbootest from the Boosting model--&gt; accuracy = 1 -- pstackedfinal from the stacked model--&gt; accuracy =0.3684

Table below shows the votes for each of the options A...E for cases 1 through 20.

    ##     value
    ## id   A B C D E
    ##   1  1 1 2 0 0
    ##   2  3 0 1 0 0
    ##   3  0 2 2 0 0
    ##   4  3 0 1 0 0
    ##   5  3 0 1 0 0
    ##   6  0 1 0 0 3
    ##   7  1 0 2 1 0
    ##   8  0 3 1 0 0
    ##   9  4 0 0 0 0
    ##   10 4 0 0 0 0
    ##   11 1 3 0 0 0
    ##   12 0 1 3 0 0
    ##   13 0 4 0 0 0
    ##   14 4 0 0 0 0
    ##   15 0 1 0 0 3
    ##   16 0 2 0 0 2
    ##   17 4 0 0 0 0
    ##   18 0 3 0 1 0
    ##   19 4 0 0 0 0
    ##   20 0 4 0 0 0
