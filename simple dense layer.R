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


#This program continues the idea of the machine learning the correct
#weights and biases (from the linear classification program). 
#In order to classify data-points correctly
#it now uses more than one layer, so as to do more things to the data
#- a 'thing' at each layer.
#Here, it will change the weights and biases in the first layer
#and use softmax in the second layer.
#Softmax takes the answers from the first layer and puts them into
#a probability space between 0 and 1... the chance of this being a '7'
#is 0.15, etc...
#The whole program works on a data-set of numbers.
#It looks at a pixelised representation of the number and works out
#from the position of the pixels
#the probability of it being a certain number
#(it was then used as a basis for scanning zipcodes).
#When the program runs you will see 'Epochs'- this
#is the data being passed through both layers
#and the results of the loss function, which will
#mainly decrease, as the predictions get more accurate.
#Finally it prints an accuracy for the whole program.


#function setting up of a ranomiser for arrays - this will later produce the weights
random_array <- function(dim, min, max){
  array(runif(prod(dim), min, max), dim)
}

#this is the general shape of the layers
#it has weights (using the randomiser), biases, and activation function
layer_naive_dense <- function(input_size, output_size, activation){
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"
  
  self$activation <- activation
  
  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)
  
  b_shape <- c(output_size)
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)
  
  self$weights <- list(self$W, self$b)
  
  self$call <- function(inputs){
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }
  
  self
  
}

#a wrapper for the layers, above. This calls one layer after another
#and tracks the weights
naive_model_sequential <- function(layers){
  
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"
  
  self$layers <- layers
  
  weights <- lapply(layers, function(layer) layer$weights)
  self$weights <- do.call(c, weights)
  
  self$call <- function(inputs) {
    x <- inputs
    for(layer in self$layers)
      x <- layer$call(x)
    x
  }
  
  self
}

#the specific set-up of the layer
#what is coming in to each, what it expects to go out
#and the activation function that makes the
#output non-linear, including softmax at the end
model <- naive_model_sequential(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512, 
                    activation = tf$nn$relu),
  layer_naive_dense(input_size = 512, output_size = 10,
                    activation = tf$nn$softmax)
))
stopifnot(length(model$weights) == 4)

#takes the input of a batch of digitilised numbers with their labels
#from the MNIST data-set and turns it into batches
#of data to feed through the layers
new_batch_generator <- function(images, labels, batch_size = 128) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "BatchGenerator"
  stopifnot(nrow(images) == nrow(labels))
  self$index <- 1
  self$images <- images
  self$labels <- labels 
  self$batch_size <- batch_size
  self$num_batches <- ceiling(nrow(images)/batch_size)
  
  self$get_next_batch <- function(){
    start <- self$index
    if(start > nrow(images))
      return(NULL)
    
    end <- start + self$batch_size - 1
    if(end > nrow(images))
      end <- nrow(images)
    
    self$index <- end + 1
    indices <- start:end
    list(images = self$images[indices, ],
         labels = self$labels[indices])
  }
  
  self
}

#the actual passing through the layers
#uses a gradient tape- which records all the gradients
#of the loss functions, so that the weights can be adjusted
#so that the gradients are less, and when the gradient reaches zero-ish
#we are at our most accurate... because the difference between the
#predictions and the targets have been minimised
one_training_step <- function(model, images_batch, labels_batch) {
  with(tf$GradientTape() %as% tape, {
    predictions <- model$call(images_batch)
    per_sample_losses <-
      loss_sparse_categorical_crossentropy(labels_batch, predictions)
    average_loss <- mean(per_sample_losses)
  })
  gradients <- tape$gradient(average_loss, model$weights)
  update_weights(gradients, model$weights)
  average_loss
}

learning_rate <- 1e-3

update_weights <- function(gradients, weights){
  stopifnot(length(gradients) == length(weights))
  for(i in seq_along(weights))
    weights[[i]]$assign_sub(
      gradients[[i]] *learning_rate)
}

#this is a short-hand repeat of the code above
#optimizer <- optimizer_sgd(learning_rate=1e-3)

#update_weights <- function(gradients, weights)
#  optimizer$apply_gradients(zip_lists(gradients, weights))


## -------------------------------------------------------------------------
#str(zip_lists(gradients = list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3"),
#              weights = list("weight_1", "weight_2", "weight_3")))


#the function that puts it all together, pushing batches
#of images and labels into the layers so that
#the program can predict what images goes with what label
#then prints out the result
fit <- function(model, images, labels, epochs, batch_size = 128) {
  for (epoch_counter in seq_len(epochs)){
    cat("Epoch ", epoch_counter, "\n")
    batch_generator <- new_batch_generator(images, labels)
    for (batch_counter in seq_len(batch_generator$num_batches)){
      batch <- batch_generator$get_next_batch()
      loss <- one_training_step(model, batch$images, batch$labels)
      if (batch_counter %% 100 == 0)
        cat(sprintf("loss at batch %s:%.2f\n", batch_counter, loss))
    }
  }
}

#taking the dataset in. This part is why keras needs to be installed
mnist <- dataset_mnist()
train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28))/255
test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28))/255
test_labels <- mnist$test$y
train_labels <- mnist$train$y

#calls the program - bit like 'main'
fit(model, train_images, train_labels, epochs = 10, batch_size = 128)

#finds the accuracy and prints
predictions <- model$call(test_images)
predictions <- as.array(predictions)
predicted_labels <- max.col(predictions) - 1
matches <- predicted_labels == test_labels
cat(sprintf("accuracy: %.2f\n", mean(matches)))