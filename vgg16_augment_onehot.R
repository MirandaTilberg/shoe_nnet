library(magrittr)
library(keras)
use_backend("tensorflow")
#install_keras()

classes <- c("bowtie", "chevron", "circle", "hexagon", "line",
             "pentagon", "quad", "star", "text", "triangle")

start_date <- Sys.Date()

name.file <- function(date, ext) {
  work_dir <- #this will be the folder to save the models after running
  path <- paste(work_dir, date, sep = "/")
  
  pretrained_base = "vgg16"
  mod_type = "onehotaug"
  nclass = paste0(length(classes), "class")
  pixel_size = "256"
  
  filename <- paste(date, pretrained_base, mod_type, nclass, 
                    pixel_size, sep = "_")
  
  return(paste(path, filename, ext, sep = ""))
}

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256, 256, 3)
)

# This needs to be the home of the three image directories
base_dir <- #file.path("/models/R_proccessed_images", start_date)
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))

library(stringr)
fnames <- list.files(train_dir)
fnames.front <- str_remove(fnames, ".jpg")
img.loc <- file.path(train_dir, fnames)

library(jpeg)

for (i in 1:n_train) {
  
  img <- readJPEG(img.loc[i])
  dim(img) <- c(1, dim(img))
  
  gen_images <- image_data_generator(rotation_range = 40,
                                     width_shift_range = 0.05,
                                     height_shift_range = 0.05,
                                     shear_range = 0.4,
                                     zoom_range = 0.1,
                                     horizontal_flip = TRUE)
  
  images_iter <- flow_images_from_data(
    x=img, y=NULL,
    generator=gen_images,
    batch_size=1,
    save_to_dir=train_aug_dir,
    save_prefix=paste("aug", fnames.front[i], sep="_"),
    save_format="jpeg"
  )
  
  reticulate::iter_next(images_iter)
  reticulate::iter_next(images_iter)
  reticulate::iter_next(images_iter)
}

extract_features2 <- function(directory, sample_count) {
  library(jpeg)
  
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


png(name.file(start_date, ".png"))
plot(history)
dev.off()

save_model_hdf5(model, name.file(start_date, ".h5"))

preds <- model %>% predict(test$features)
test_labs <- test$labels
colnames(preds) <- colnames(test_labs) <- classes

save(preds, test_labs, file = name.file(start_date, ".Rdata"))
# base::save.image(name.file(start_date, ".RData"))
