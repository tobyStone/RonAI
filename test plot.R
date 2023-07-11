num_samples_per_class <- 1000
Sigma <- rbind(c(1, 0.5), c(0.5, 1))
negative_samples <- MASS::mvrnorm(n = num_samples_per_class, mu = c(0, 3), Sigma = Sigma)
positive_samples <- MASS::mvrnorm(n = num_samples_per_class, mu = c(3, 0), Sigma = Sigma)
inputs <- rbind(negative_samples, positive_samples)
targets <- rbind(array(0, dim = c(num_samples_per_class, 1)), array(1, dim = c(num_samples_per_class, 1)))
plot(x = inputs[,1], y = inputs[,2], col = ifelse(targets[,1] == 0, "purple", "green"))


