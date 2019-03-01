dataset = read.csv('d:\\GITHUB\\Machine_learning\\03_Multiple_Linear_Regression\\50_Startups.csv')

# dataset = daatset[, 2:3]


dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

regressor = lm(formula = Profit ~ ., data = training_set)

y_pred = predict(regressor, newdata = test_set)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)
# ----------------------

library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$R.D.Spend, y = training_set$Profit),
             colour = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor, newdata = training_set)),
            colour = 'green') 

ggplot() + 
  geom_point(aes(x = test_set$R.D.Spend, y = test_set$Profit),
             colour = 'red') +
  geom_line(aes(x = training_set$R.D.Spend, y = predict(regressor, newdata = training_set)),
            colour = 'green') 

