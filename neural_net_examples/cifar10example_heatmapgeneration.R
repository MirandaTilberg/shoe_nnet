library(keras)
plot_heatmap <- function(heatmap, width = 224, height = 224,
                         bg = "white", col = terrain.colors(12)) {
  op = par(mar = c(0,0,0,0))
  on.exit({par(op)}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}


cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)


N <- 48
idx_start <- (N - 32)/2 + 1
idx <- idx_start:(idx_start + 31)

# Select target image and format for prediction
#### Pretrained classifier requires 224x224
#### New classifier will take image size as is (NxN)

x_train_N <- array(0, dim = c(50000, N, N, 3))
x_train_N[, idx, idx, ] <- x_train[1:50000,,,]
y_train_N <- y_train[1:50000,]

x_test_N <- array(0, dim = c(10000, N, N, 3))
x_test_N[, idx, idx, ] <- x_test[1:10000,,,]
y_test_N <- y_test[1:10000,]

img_N <- x_test_N[1,,,]
dim(img_N) <- c(1, N, N, 3)

input <- layer_input(shape = c(N, N, 3))

# Load VGG16 base (without classifier)
VGG16_base <- application_vgg16(
  include_top = FALSE,
  input_tensor = input
)

output <- VGG16_base$output %>%
  layer_flatten(input_shape = layer_input(shape = c(N, N, 512))) %>%
  layer_dense(units = 10, activation = "softmax")

model <- keras_model(input, output)

freeze_weights(model, from = 1, to = 'block5_conv3')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

model %>% fit(
  x_train_N, y_train_N,
  batch_size = 12,
  epochs = 1,
  validation_data = list(x_test_N, y_test_N),
  shuffle = TRUE
)

predictions <- model %>% predict(img_N)

img_output <- model$output[,which.max(predictions)]

last_conv_layer <- model %>% get_layer("block5_conv3")
grads <- k_gradients(img_output, last_conv_layer$output)[[1]]

# finish making the heatmap
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,]))
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img_N))
for (i in 1:512) {
  conv_layer_output_value[,,i] <-
    conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap / max(heatmap)

plot_heatmap(heatmap, width = N, height = N)

save_model_hdf5(model, "/home/tiltonm/shoe_nnet/FAPI_testmodel.h5")
model <- load_model_hdf5("/home/tiltonm/shoe_nnet/FAPI_testmodel.h5")
