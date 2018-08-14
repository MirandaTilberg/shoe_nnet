library(magrittr)
library(keras)
use_backend("tensorflow")
#install_keras()

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256, 256, 3)
)

summary(conv_base)


base_dir <- "~/shoes/circletriangle"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

datagen <- image_data_generator(rescale = 1/255)
batch_size <- 1

extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 8, 8, 512))  
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(256, 256),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}
n_train <- length(list.files(file.path(train_dir, "triangle"))) + length(list.files(file.path(train_dir, "circle")))
n_validation <- length(list.files(file.path(validation_dir, "triangle"))) + length(list.files(file.path(validation_dir, "circle")))
n_test <- length(list.files(file.path(test_dir, "triangle"))) + length(list.files(file.path(test_dir, "circle")))


train <- extract_features(train_dir, n_train)
validation <- extract_features(validation_dir, n_validation)
test <- extract_features(test_dir, n_test)

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 8 * 8 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)


model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 8 * 8 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)

plot(history)

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

cat("This is the number of trainable weights before freezing",
    "the conv base:", length(model$trainable_weights), "\n")
freeze_weights(conv_base)

cat("This is the number of trainable weights after freezing",
    "the conv base:", length(model$trainable_weights), "\n")


train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(256, 256),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(256, 256),
  batch_size = 20,
  class_mode = "binary"
)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

save_model_hdf5(model, "circle_and_triangle_small_3.h5")
summary(conv_base)
unfreeze_weights(conv_base, from = "block3_conv1")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

save_model_hdf5(model, "circle_and_triangle_small_4.h5")
plot(history)

test_generator <- flow_images_from_directory(
  test_dir, 
  test_datagen, 
  target_size = c(256, 256),
  batch_size = 20, 
  class_mode = "binary"
)

model <- load_model_hdf5("circle_and_triangle_small_4.h5")
model %>% evaluate_generator(test_generator, steps = 50)
