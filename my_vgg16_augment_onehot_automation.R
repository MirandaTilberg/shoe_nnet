library(magrittr)
library(lubridate)
library(stringr)
library(jpeg)
library(keras)

use_backend("tensorflow")
#install_keras()

classes <- c("bowtie", "chevron", "circle", "hexagon", "line",
             "pentagon", "quad", "star", "text", "triangle")

work_dir <- "/models/shoe_nn/TrainedModels"
start_date <- Sys.Date()

processed_img_folder <- list.files("/models/shoe_nn/RProcessedImages/") %>% 
  as_datetime() %>%
  max(na.rm = T) %>%
  gsub("[^0-9\\ ]", "", .) %>%
  gsub(" ", "-", .)

model_dir <- file.path(work_dir, processed_img_folder)
dir.create(model_dir)

name_file <- function(date, ext) {
  pretrained_base = "vgg16"
  mod_type = "onehotaug"
  nclass = paste0(length(classes), "class")
  pixel_size = "256"
  
  filename <- paste(date, pretrained_base, mod_type, 
                    nclass, pixel_size, sep = "_")
  
  file.path(model_dir, filename) %>%
    paste0(., ext)
}

base_dir <- file.path("/models/shoe_nn/RProcessedImages", processed_img_folder)
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")



n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))

img_names <- list.files(train_dir) %>% str_remove(., ",jpg")
img_loc <- list.files(train_dir, full.names = T)

### if we decide to reaugment, this will remove augmented images
# file.remove(list.files(train_dir, pattern = "aug_", full.names = T))

### if we keep previously augmented images, run this as-is
if (length(list.files(train_dir, pattern = "aug_")) <= 2000) {
  
  for (i in 1:n_train) {
    
    img <- readJPEG(img_loc[i])
    dim(img) <- c(1, dim(img))
    
    aug_generator <- image_data_generator(rotation_range = 40,
                                          width_shift_range = 0.05,
                                          height_shift_range = 0.05,
                                          shear_range = 0.4,
                                          zoom_range = 0.1,
                                          horizontal_flip = TRUE)
    
    images_iter <- flow_images_from_data(
      x = img, y = NULL,
      generator = aug_generator,
      batch_size = 1,
      save_to_dir = train_aug_dir,
      save_prefix = paste("aug", img_names[i], sep="_"),
      save_format = "jpeg"
    )
    
    reticulate::iter_next(images_iter)
    reticulate::iter_next(images_iter)
    reticulate::iter_next(images_iter)
  }
  
}

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256, 256, 3)
)

extract_features2 <- function(directory, sample_count) {
  features <- array(0, dim = c(sample_count, 8, 8, 512))  
  labels <- array(0, dim = c(sample_count, length(classes)))
  
  files <- list.files(directory)
  
  for (i in 1:sample_count) {
    
    fname <- files[i]
    str <- substr(fname, 1, regexpr("-",fname)-1)
    
    for (j in 1:length(classes)) {
      labels[i, j] <- grepl(classes[j], str)
    }
    
    img <- readJPEG(file.path(directory, files[i]))
    dim(img) <- c(1, 256, 256, 3)
    features[i,,,] <- conv_base %>% predict(img)
  }
  
  list(
    features = features, 
    labels = labels
  )
}

train <- extract_features2(train_dir, n_train)
validation <- extract_features2(validation_dir, n_validation)
test <- extract_features2(test_dir, n_test)


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
  layer_dense(units = length(classes), activation = "sigmoid")

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


png(name_file(start_date, ".png"))
plot(history)
dev.off()

save_model_hdf5(model, name_file(start_date, ".h5"))

preds <- model %>% predict(test$features)
test_labs <- test$labels
colnames(preds) <- colnames(test_labs) <- classes

save(classes, preds, test_labs, file = name_file(start_date, ".rdata"))
# base::save.image(name_file(start_date, ".rdata"))
