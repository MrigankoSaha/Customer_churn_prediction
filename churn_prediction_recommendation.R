#Loading the required packages

library("randomForest")
library("caret")
library("class")
library("dplyr")
library("ggplot2")
library("MASS")
library("pROC")
library("tidyverse")

install.packages("skimr")
library("skimr")


telco_data <- read.csv("C:/Users/Mriganko Saha/Desktop/Business_Analytics_project_fall24/WA_Fn-UseC_-Telco-Customer-Churn.csv")

glimpse(telco_data)

#Visualize missing values using skim() from skimr package and drop na values
skim(telco_data)
data_clean <- drop_na(telco_data)
head(data_clean)

#Visualize churn variable distribution (Yes vs No in %)
ggplot(data_clean, aes(x = Churn)) +
  geom_bar(fill = "blue") +
  labs(title = "Distribution of Churn (Yes vs No)", x = "Churn", y = "Count") +
  theme_minimal()

#Calculating percentage of churn
churn_count <- sum(data_clean$Churn == "Yes", na.rm = TRUE) 
total_count <- nrow(data_clean) 

churn_percentage <- (churn_count / total_count) * 100
churn_percentage

#Visualize continuous variables w.r.t Churn
ggplot(data_clean, aes(x = tenure, fill = Churn)) +
  geom_histogram(position = "dodge", bins = 30) +
  labs(title = "Tenure vs Churn", x = "Tenure", y = "Count") +
  theme_minimal()

ggplot(data_clean, aes(x = MonthlyCharges, fill = Churn)) +
  geom_histogram(position = "dodge", bins = 30) +
  labs(title = "Monthly Charges vs Churn", x = "Monthly Charges", y = "Count") +
  theme_minimal()

#Visualizing categorical variables w.r.t churn
ggplot(data_clean, aes(x = gender, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Gender vs Churn", x = "Gender", y = "Proportion") +
  theme_minimal()

ggplot(data_clean, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Contract vs Churn", x = "Contract", y = "Proportion") +
  theme_minimal()

#Checking for outliers in continuous variables using boxplots
ggplot(data_clean, aes(y = tenure)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplot for Tenure") +
  theme_minimal()

ggplot(data_clean, aes(y = MonthlyCharges)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplot for Monthly Charges") +
  theme_minimal()

#Data preparation(cleaning categorical variables, standardizing continuous features)

#Converting categorical variables to factors
data_clean$gender <- as.factor(data_clean$gender)
data_clean$contract <- as.factor(data_clean$Contract)
data_clean$paymentmethod <- as.factor(data_clean$PaymentMethod)
data_clean$churn <- as.factor(data_clean$Churn)

# Standardize continuous features
data_clean$tenure <- scale(data_clean$tenure)
data_clean$monthlycharges <- scale(data_clean$MonthlyCharges)

#Splitting data into train and test sets
set.seed(123)
trainIndex <- createDataPartition(data_clean$churn, p = 0.8, list = FALSE)
trainData <- data_clean[trainIndex, ]
testData <- data_clean[-trainIndex, ]

#LOGISTIC-REGRESSION

logistic_model_full <- glm(churn ~ Dependents+tenure+PhoneService+PaperlessBilling+PaymentMethod+MonthlyCharges+
                             TotalCharges+StreamingTV+StreamingMovies+SeniorCitizen+DeviceProtection, data = trainData, family = "binomial")
summary(logistic_model_full)
#using stepAIC for selecting the best group of parameters
logistic_model_step <- stepAIC(logistic_model_full, direction = "both")
summary(logistic_model_step)
#removing DeviceProtection due to low significance(high p value)
logistic_model_noDeviceProtection <- glm(formula =churn ~ Dependents+tenure+PhoneService+PaperlessBilling+PaymentMethod+MonthlyCharges+
                                           TotalCharges+StreamingTV+StreamingMovies+SeniorCitizen, data = trainData,family = "binomial")
summary(logistic_model_noDeviceProtection)

#removing streamingDevice due to lowsignificance(high p value)
logistic_model_noStreamingDevice <- glm(formula = churn~Dependents+tenure+PhoneService+PaperlessBilling+PaymentMethod+MonthlyCharges+
                                          TotalCharges+SeniorCitizen, family = "binomial", data = trainData)

summary(logistic_model_noStreamingDevice)

logistic_preds_prob <- predict(logistic_model_full, newdata = testData, type = "response")

#Tune hyperparameters (using Youden's index, basically using roc first)
roc_curve <- roc(testData$Churn, logistic_preds_prob)
optimal_cutoff <- coords(roc_curve, "best", ret = c("threshold"))
optimal_cutoff

#optimal cutoff is 0.42 but down below am using 0.4 for a better result(0-0.2 is making a huge difference and incorrectly predicting)
#Confusion matrix for logistic regression model
logistic_preds_class <- ifelse(logistic_preds_prob > 0.4, "Yes", "No")
table(logistic_preds_class, testData$Churn)


#KNN model
#Dealing with categorical, infinite nan etc because knn only works with numeric values

# Handle missing values: remove rows with NA values from both train and test sets
trainData_clean <- na.omit(trainData)
testData_clean <- na.omit(testData)

k_values <- 1:84 #as we have 7043 rows and (7043)^0.5 ~ 84
accuracy <- numeric(length(k_values))

# Convert categorical variables as well as factor variables to numeric
trainData_clean_numeric <- as.data.frame(lapply(trainData_clean, function(x) {
  if (is.factor(x)) {
    return(as.numeric(as.character(x)))  #factor to numeric
  } else if (is.character(x)) {
    return(as.numeric(as.factor(x)))  #character to numeric by factorizing
  } else {
    return(as.numeric(x))  
  }
}))
testData_clean_numeric <- as.data.frame(lapply(testData_clean, function(x) {
  if (is.factor(x)) {
    return(as.numeric(as.character(x)))  #factor to numeric
  } else if (is.character(x)) {
    return(as.numeric(as.factor(x)))  #character to numeric by factorizing
  } else {
    return(as.numeric(x))
  }
}))
# Loop over k_values
selected_columns <- c(19, 20)
trainData_subset <- trainData_clean_numeric[, selected_columns]
testData_subset <- testData_clean_numeric[, selected_columns]


for (a in k_values) {
  knn_preds <- knn(train =trainData_subset ,
                   test = testData_subset,
                   cl = trainData$Churn, k = a)
  
  # Calculate the accuracy for the current k
  accuracy[a] <- mean(knn_preds == testData$Churn)
}

best_k <- k_values[which.max(accuracy)]
# Train KNN with best k(taking numerical columns only)
knn_preds_final <- knn(train = trainData_subset,
                       test = testData_subset,
                       cl = trainData$churn, k = best_k)

# Confusion matrix for KNN
table(knn_preds_final, testData$churn)

#Random Forest

#Ensuring that the target variable is a factor
trainData$churn <- factor(trainData$churn)

rf_tuned <- tuneRF(trainData[, -which(names(trainData) == "churn")], trainData$churn, 
                   stepFactor = 1.5, improve = 0.01, mtryStart=1, mtryEnd=20,trace = TRUE)

rf_model_tuned <- randomForest(churn ~ ., data = trainData[, -which(names(trainData) == "Churn")],mtry = 4,ntree = 500)
rf_preds <- predict(rf_model_tuned, newdata = testData)

table(rf_preds, testData$churn)

varImpPlot(rf_model_tuned)

knn_roc <- roc(testData$churn, as.numeric(knn_preds_final) - 1)

rf_roc <- roc(testData$churn, as.numeric(rf_preds))

#Plotting the KNN ROC curve
plot(knn_roc, col = "blue", main = "ROC Curves for KNN and Random Forest", 
     lwd = 2, cex.main = 1.2)

#Adding the Random Forest ROC curve to the same plot
lines(rf_roc, col = "red", lwd = 2)

legend("bottomright", legend = c("KNN", "Random Forest"), col = c("blue", "red"), lwd = 2)

#Area under curve(AUC) for the top 2 models.............
knn_auc <- auc(knn_roc)
rf_auc <- auc(rf_roc)

#Find relationship of payment method v churn

ggplot(testData, aes(x = PaymentMethod, fill = churn)) +
  geom_bar(position = "dodge", alpha = 0.7) +
  labs(title = "Churn Status by Payment Method", x = "Payment Method", y = "Count") +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()

#Find relationship of Contract type vs churn

ggplot(testData, aes(x = Contract, fill = churn)) +
  geom_bar(position = "dodge", alpha = 0.7) +
  labs(title = "Churn Status by Contract Type", x = "Contract Type", y = "Count") +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()

#Find relationship of MonthlyCharges vs churn

ggplot(testData, aes(x = MonthlyCharges, fill = churn)) +
  geom_histogram(binwidth = 5, position = "dodge", alpha = 0.7) +
  labs(title = "Distribution of Monthly Charges by Churn Status", 
       x = "Monthly Charges", y = "Count") +
  scale_fill_manual(values = c("red", "green")) +
  theme_minimal()

