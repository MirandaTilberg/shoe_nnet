library(magrittr)
library(keras)
use_backend("tensorflow")
#install_keras()

gpu <- F
classes <- c("bowtie", "chevron", "circle", "hexagon",
             "quad", "star", "text", "triangle")

load("~/shoe_nnet/shoe_models/OneHot/090918_vgg16_onehot_8class_256.RData")

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256, 256, 3)
)

base_dir <- ifelse(gpu, "/work/CSAFE/shoes/onehot", "~/shoe_nnet/shoes/onehot")
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))



dim(save_train$features)


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

extract_data <- function(directory, sample_count, verbose = F) {
  #library(imager)
  library(jpeg)
  
  img_size <- c(256, 256, 3)
  data <- array(0, dim = c(sample_count, img_size))
  
  files <- list.files(directory)
  for (i in 1:sample_count) {
    if (verbose) cat(paste(i, ", ", sep=""))
    img <- readJPEG(file.path(directory, files[i]))
    dim(img) <- c(1, img_size)
    data[i,,,] <- img
  }
  
  data
  
}

train_aug <- extract_data(train_dir, n_train, verbose = T)

images_iter <- flow_images_from_data(
  x=train_aug, y=train$labels,
  generator=train_datagen,
  batch_size=9#,
  #save_to_dir=images_dir,
  #save_prefix="aug",
  #save_format="jpeg"
)


train_generator <- flow_images_from_data(train_aug, train$labels)

test_datagen <- image_data_generator(rescale = 1/255)


model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 8 * 8 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = length(classes), activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# history <- model %>% fit_generator(
#   train_generator,
#   steps_per_epoch = 100,
#   epochs = 30,
#   validation_data = list(validation$features, validation$labels),
#   validation_steps = 50
)

history <- model %>% fit(
  flow_images_from_data(train_aug, train$labels),
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)


