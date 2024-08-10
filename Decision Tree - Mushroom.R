# Anca Pitigoi - 08/07/2024 

#upload packages

library(pacman)
pacman::p_load(rpart, caret, rpart.plot, rattle, readxl,
  readr, janitor, ggplot2, dplyr, gt, tidyr, ggthemes, scales , gridExtra, corrplot)

# quietly = TRUE does not seem to work with p_load() to suppress the loading messages


                            #### 1. Loading the dataset ####

original <- read_excel("")
# This function loads the excel file. The Import Data set from the global environment can also
  # be used. It is basically the same function. It is fast and efficient even when the files
  # are large.

mushrooms <- original # Keeping the original data set


                            #### 2. Data Cleaning ####

str(mushrooms)
# There are 8,124 observations & 23 character features.


# Number of rows with missing values
nrow(mushrooms) - sum(complete.cases(mushrooms))
# There are no missing observations in the data set.


# Cleaning the names
mushrooms <- clean_names(mushrooms) 
# Data set loaded differently than in the given code, so names had to be cleaned to support
  # the functions in R


# Deleting redundant variable "veil_type"
mushrooms$veil_type <- NULL
# This observation was eliminated because all the mushrooms have the same type of veil


skimr::skim(original) #this function helps to see this redundancy bcs it shows 
  # how many unique observations there are


                            #### 3. EDA ####

# Analyzing the odor variable
table(mushrooms$class,mushrooms$odor) # columns represent the odors and rows tell whether
# the mushroom is poisonous or edible. The function makes a table and counts the observations.
# Based on the other, it can be seen which mushrooms are poisonous.


# Analyzing class-feature combinations
number.perfect.splits <- apply(X=mushrooms[-1], MARGIN = 2, FUN = function(col){
  t <- table(mushrooms$class,col)
  sum(t == 0)
})
# The function firstly tells R to input the mushrooms data set without the class variable.
# After that MARGIN = 2 means that the function is applied over columns.
# FUN = function(col) tells this function to apply for each column.
# The inner part of the function shows how many observations in each column did 
# not occur for the class of the mushroom. This is perfect to show which features
# should be looked at when picking mushrooms.


# Descending order of perfect splits
order <- order(number.perfect.splits, decreasing = TRUE) # descending order is calculated 
number.perfect.splits <- number.perfect.splits[order] # reorders the features based on
  # the previous step


# Plot graph
par(mar=c(10,2,2,2)) # sets the edges of the plot
barplot(number.perfect.splits,
        main="Number of perfect splits vs feature",
        xlab="",ylab="Feature",las=2,col="wheat")
# mushrooms can be mostly differentiated by odor, stalk color above the ring or below 
  # the ring, and by the spore print color.


                      #### 4. Build Decision Tree ####


# Data splicing
set.seed(12345) # facilitates reproducible results
train <- sample(1:nrow(mushrooms), size = ceiling(0.80*nrow(mushrooms)), replace = FALSE)
# generates random numbers until 6500 (80% of data set), and sampling is done 
  # without replacement. Ceiling rounds up the value.

# Training set
mushrooms_train <- mushrooms[train,] # selects 80% of obs. from the original data set
  # according to the index created in train vector

# Test set
mushrooms_test <- mushrooms[-train,] # selects the remaining 20% of the rows


# Penalty matrix
penalty.matrix <- matrix(c(0,1,10,0), byrow=TRUE, nrow=2) # creates a penalty matrix 
  # like a confusion matrix, because some types of errors have different costs 
  # associated with them. 
  # 0 = True Positive & True Negative, no penalty
  # 1 = False Negative, predict the mushroom is poisonous when it is edible
  # 10 = False Positive, predict the mushroom is edible when it is poisonous, this
    # would be the worst case scenario to happen.


# Building the classification tree with rpart
tree <- rpart(class~., #uses all predictors in the training set
              data = mushrooms_train,
              parms = list(loss = penalty.matrix), #minimizes the chances for False Positive
              method = "class") #this is a classification model
# printing the tree in a text format will be difficult to analyze, so the next
  # plot will simplify the interpretation


# Visualize the decision tree with rpart.plot
rpart.plot(tree, nn=TRUE) #nn=T shows the # obs. in each node



                    #### 5. Tuning the tree ####

# Choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]), "CP"]

tree$cptable
# 0.01 is the complexity parameter with the lowest cross-validated error


# Tree prunning using the best complexity parameter
tree <- prune(tree, cp=cp.optim) # prunes the tree based on 0.01 CP

rpart.plot(tree, nn=TRUE) # the tree is unchanged because we already had the best
# option.



                #### 6. Validating the Decision Tree ####

#Testing the model
pred <- predict(object = tree, # specifies the decision tree to use
                mushrooms_test[-1], # removes the class feature
                type = "class") # specifies is a classification


# Calculating accuracy
t <- table(mushrooms_test$class, pred) # prepares the data for the confusion matrix

confusionMatrix(t) # shows how valid the model is
# Seems like the model is incredibly accurate at predicting poisonous and edible 
  # mushrooms.
# The model predicts with 100% accuracy, this considered too good to be true.



########## Don't apply the penalty matrix ###########


tree2 <- rpart(class~., #uses all predictors in the training set
              data = mushrooms_train, method = "class")

# Visualize the decision tree with rpart.plot
rpart.plot(tree2, nn=TRUE) #nn=T shows the # obs. in each node

# Only two levels are left, and two features are important: odor and spore print color


# Choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree2$cptable[which.min(tree2$cptable[,"xerror"]), "CP"]

tree2$cptable
# 0.01 is the complexity parameter with the lowest cross-validated error; 2 splits


# Tree prunning using the best complexity parameter
tree2 <- prune(tree2, cp=cp.optim) # prunes the tree based on 0.01 CP

rpart.plot(tree2, nn=TRUE) # the tree is unchanged because we already had the best
# option.



#Testing the model
pred2 <- predict(object = tree2, # specifies the decision tree to use
                mushrooms_test[-1], # removes the class feature
                type = "class") # specifies is a classification


# Calculating accuracy
t2 <- table(mushrooms_test$class, pred2) # prepares the data for the confusion matrix

confusionMatrix(t2) # shows how valid the model is pretty accurate
# The model predicts with 99% accuracy, this considered too good to be true, however
  # there are 11 poisonous mushrooms that were labeled as edible.

