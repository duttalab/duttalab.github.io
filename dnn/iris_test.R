library(kerasR)
library(keras)
install.packages('devtools')
devtools::install_github("statsmaths/kerasR")
install_tensorflow()
use_condaenv("r-tensorflow")
# Make your dummy data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)

# Make dummy target values for your dummy data
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)

names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")
cor(iris$Petal.Length, iris$Petal.Width)
# Store the overall correlation in `M`
M <- cor(iris[,1:4])

# Plot the correlation plot with `M`
corrplot(M, method="circle")

# Build your own `normalize()` function
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# Normalize the `iris` data
iris_norm <- as.data.frame(lapply(iris[1:4], normalize))

# Return the first part of `iris` 
head(iris)
iris[,5] <- as.numeric(iris[,5]) -1

# Turn `iris` into a matrix
iris <- as.matrix(iris)

# Set iris `dimnames` to `NULL`
dimnames(iris) <- NULL

# Normalize the `iris` data
iris <- normalize(iris[,1:4])

# Return the summary of `iris`
summary(iris)

# Determine sample size
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

# Split the `iris` data
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]

# Split the class attribute
iris.trainingtarget <- iris[ind==1, 4]
iris.testtarget <- iris[ind==2, 4]

iris.trainLabels <- to_categorical(iris.trainingtarget)

# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)
iris.trainLabels <- to_categorical(iris.trainingtarget)
iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)

iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)

# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')
# Print a summary of a model
summary(model)

# Get model configuration
get_config(model)

# Get layer configuration
get_layer(model, index = 1)

# List the model's layers
model$layers

# List the input tensors
model$inputs

# List the output tensors
model$outputs

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model 
model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)

# Store the fitting history in `history` 
history <- model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2
)

# Plot the history
plot(history)

?dataset_cifar10 #to see the help file for details of dataset
cifar<-dataset_cifar10()
