#### cifar10 example modified from
#### https://keras.rstudio.com/articles/examples/cifar10_cnn.html

#### Heatmap code modified from Ch 3 of Deep Learning with R
#### by Francois Challet with J.J. Allaire.
#### For more info, see https://www.manning.com/books/deep-learning-with-r

library(keras)

# Data Preparation --------------------------------------------------------

cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs  
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)

# Select target image and format for prediction
#### Pretrained classifier requires 224x224
#### New classifier will take image size as is (32x32)
img_32 <- x_test[1,,,]
dim(img_32) <- c(1, 32, 32, 3)
img_224 <- array(0, dim = c(1, 224, 224, 3))
img_224[1, 97:128, 97:128, ] <- img_32


# Heatmap using pre-trained classifier -------------------------------------

# Load VGG16 (including classifier)
VGG16 <- application_vgg16(
  include_top = TRUE,
  weights = "imagenet"
)

predictions <- VGG16 %>% predict(img_224)
imagenet_decode_predictions(predictions, top = 3)[[1]]

img_output <- VGG16$output[,which.max(predictions)]
last_conv_layer <- VGG16 %>% get_layer("block5_conv3")
grads <- k_gradients(img_output, last_conv_layer$output)[[1]]
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
iterate <- k_function(list(VGG16$input),
                      list(pooled_grads, last_conv_layer$output[1,,,]))
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img_224))
for (i in 1:512) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
heatmap <- apply(conv_layer_output_value, c(1,2), mean)
heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)

plot_heatmap <- function(heatmap, width = 224, height = 224,
                         bg = "white", col = terrain.colors(12)) {
  op = par(mar = c(0,0,0,0))
  on.exit({par(op)}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

plot_heatmap(heatmap) # Desired outcome


# Create and train model w/ VGG16 base, new classifier --------------------
#### Note: the training parameters in this example are optimized for speed
#### and not for classification accuracy

# Load VGG16 base (without classifier)
VGG16_base <- application_vgg16(
  include_top = FALSE,
  weights = "imagenet",
  input_shape = c(32, 32, 3)
)

# Build classifier on VGG16 base
model <- keras_model_sequential() %>%
  VGG16_base %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = 1*1*512) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 10, activation = "softmax")

# Freeze weights for training
freeze_weights(VGG16_base)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 1,
  validation_data = list(x_test, y_test),
  shuffle = TRUE
)

# Try to make heatmap, as before -------------------------------------------

# Get predictions, as before
predictions <- model %>% predict(img_32)
img_output <- model$output[,which.max(predictions)]

# Get output from last convolutional layer of original VGG16, as before
#### (Since weights were frozen during training, this should be
#### equivalent to taking the same layer the model we trained)
last_conv_layer <- VGG16 %>% get_layer("block5_conv3")


# This is where the trouble starts.
#### After running the next line, we get grads is NULL.
#### None of the following lines can be run.
grads <- k_gradients(img_output, last_conv_layer$output)[[1]]

# # This would finish making the heatmap
# pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
# iterate <- k_function(list(VGG16$input),
#                       list(pooled_grads, last_conv_layer$output[1,,,]))
# c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img_32))
# for (i in 1:512) {
#   conv_layer_output_value[,,i] <- 
#     conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
# }
# heatmap <- apply(conv_layer_output_value, c(1,2), mean)
# heatmap <- pmax(heatmap, 0) 
# heatmap <- heatmap / max(heatmap)
# 
# plot_heatmap(heatmap, width = 32, height = 32)