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

#a program to classify a movie review as positive or negative
#the machine learning will 'see' numbers that represent words in an
#abridged dictionary and whether they are positive or negative
#and use the relations between these facts and their position
#in the review to train
#and compare with the label of overall positive or negative review
#that has already been human-adjudicated



#pre-processed dataset of imdb reviews
#the most common 10,000 words turned into integers
#so that they can be fed into tensors (n-dimensional matrix/vector/scalar)
#and then wrangled through deep learning
#the %<% is a pipe operator that shortens the amount of code needed
#like unpacking tuples in Python
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

#integers represent words in the next line
str(train_data)
#integers 0 and 1 represent negative and positive (respectively) labels to these words
str(train_labels)

#turning the numbered words into tensors so they can be processed
#one dimension is 10,000 because there are 10,000 possible words
#a '1' will be in the position if that word is there
#the other dimension is the length of the review
#reasonably large tensors!
vectorize_sequences <- function(sequences, dimension = 10000)
{results <- array(0, dim = c(length(sequences), dimension))
for (i in seq_along(sequences)) {
  sequence <- sequences[[i]]
  for (j in sequence)
    results[i,j] <- 1
}
results
}

#turning the datasets into these tensors
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

str(x_train)

#so they can be included in the training, the labels have to be numbers, too
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)


#the model uses relu, which makes the results non-linear
#and more of a probability 'landscape'
#and sigmoid to force them into a 0 - 1
#probability
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

#initialise the model, rmsprop is how
#it learns (it takes slices of gradients, finds their mean
#in a row, squares this for positive, takes the root,
#then uses the relationship between these proportionalised-gradients
#to decide how much to change the weights by, and
#thence try to reduce the distance between targets and predictions)
#cross-entropy is the difference between expected probability
#distribution and what has come out of the layers
#binary means there are two distributions being compared
#accuracy is how the model is judged, here;
#percentage of correct predictions
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")

#training and separating chunks
#of 10,000 samples to test the others with
x_val <- x_train[seq(10000), ]
partial_x_train <- x_train[-seq(10000), ]
y_val <- y_train[seq(10000)]
partial_y_train <- y_train[-seq(10000)]

#calling fit, which is like 'main'
#training and feeding results in a history
#which can then be plotted later
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

str(history$metrics)

plot(history)


model %>% predict(x_test)