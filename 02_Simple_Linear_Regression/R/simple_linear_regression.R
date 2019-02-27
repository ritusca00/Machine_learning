#d:\GITHUB\Machine_learning\02_Simple_Linear_Regression\Salary_Data.csv
dataset = read.csv('d:\\GITHUB\\Machine_learning\\02_Simple_Linear_Regression\\Salary_Data.csv')

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

y_pred = predict(regressor, newdata = test_set)

#install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'green') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') + 
  ylab('Salary')

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'green') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') + 
  ylab('Salary')






