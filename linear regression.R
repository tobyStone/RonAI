#set up
#devtools::install_github("rstudio/tensorflow")
#library(tensorflow)
#install_tensorflow()
#miniconda install; the next is a test if tensorflow has properly installed
#it will also highlight how I'm not using a GPU/TPU
#hello <- tf$constant("Hello")
#print(hello)
#install.packages("keras")
#library(keras)
#this is where the boston housing dataset is stored
#install.packages("mlbench")
#library(mlbench)


#until now, programs have found a value and decided is it one thing or another
#based on the learning the model had been through. The lines the models have
#plotted have been an afterthought, based on where those discrete values
#had been located.
#Linear regression is all about the line.
#It uses the line to predict values that don't exist in the dataset
#either using interpolation or extrapolation. For instance: finding out stock
#prices next week, temperatures tomorrow, or house prices in a year or place not previously
#collected for.
#This is the last, and uses crime/property tax rate/other suburb prices to predict an
#unpriced suburb.


boston <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% boston

str(train_data)

#the data has too much 'spread' and different means
#this standardises it so as to avoid
#wild gradient changes
mean <- apply(train_data, 2, mean)
sd <- apply(train_data, 2, sd)

train_data <- scale(train_data, center = mean, scale = sd)
test_data <- scale(test_data, center = mean, scale = sd)


#the only new part of this model is 'mae'
#in metrics... this is:Mean Absolute Error
#and takes the average of this difference between
#predictions and targets
build_model <- function(){
  
  model <- keras_model_sequential() %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(64, activation = "relu") %>%
    layer_dense(1)
  
  model %>% compile(optimizer = "rmsprop",
                    loss = "mse",
                    metrics = "mae")
  
  model
  
}

#there is not enough data, so validation is difficult
#the validation set will be small and may have large variance
#training in Deep Learning is split in three parts:
#general training, validating this training, and a final test set;
#the data is then split into three, and there isn't
#enough of it, here
#so the training/validation third of the data is again split into 4 parts
#and each has a validation part, with the results averaged.
k <- 4
fold_id <- sample(rep(1:k, length.out = nrow(train_data)))
num_epochs <- 100
all_scores <- numeric()


for (i in 1:k) {
  cat("Processing fold#", i, "\n")
  val_indices <- which(fold_id == i)
  
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  model %>% fit (
    
    partial_train_data,
    partial_train_targets,
    epochs = num_epochs,
    batch_size = 16,
    verbose = 0
    
  )
  
  results <- model %>%
    evaluate(val_data, val_targets, verbose = 0)
  all_scores[[i]] <- results[['mae']]
  
}

all_scores

mean(all_scores)

num_epochs <- 500
all_mae_histories <- list()
for (i in 1:k) {
  cat("Processing fold#", i, "\n")
  val_indices <- which(fold_id == i)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  model <- build_model()
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 16, verbose = 0
  )
  mae_history <- history$metrics$val_mae
  all_mae_histories[[i]] <- mae_history
}

all_mae_histories <- do.call(cbind, all_mae_histories)
average_mae_history <- rowMeans(all_mae_histories)
plot(average_mae_history, xlab = "epoch", type = "l")

truncated_mae_history <- average_mae_history[-(1:10)]
plot(average_mae_history, xlab = "epoch", type = "l",
     ylim = range(truncated_mae_history))

model <- build_model()
model %>% fit(train_data, train_targets,
              epochs = 120, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)

result["mae"]

predictions <- model %>% predict(test_data)
predictions[1, ]