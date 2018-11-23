library(magrittr)
library(keras)
use_backend("tensorflow")
#install_keras()

### To train a new model, put desired classes into "classes" vector and
##### change the date in name.file() (and other variables, if appropriate)

gpu <- F
classes <- c("bowtie", "chevron", "circle", "hexagon", "line",
             "pentagon", "quad", "star", "text", "triangle")

name.file <- function(mod_num, ext) {
  work_dir <- ifelse(gpu, "/work/CSAFE/", "~/shoe_nnet/")
  mod_type_dir = "shoe_models/OneHot/"
  path <- paste(work_dir, mod_type_dir, sep = "")
  
  ### Change these if global variables change
  date = "102518_"
  pretrained_base = "vgg16_"
  mod_type = "OHAfull_"
  nclass = paste(length(classes), "class_", sep="")
  pixel_size = "256_"
  if (mod_num == 0) {
    pixel_size = "256"
    mod_num <- ""}
  
  filename <- paste(date, pretrained_base, mod_type, nclass, 
                    pixel_size, as.character(mod_num), sep = "")
  
  return(paste(path, filename, ext, sep = ""))
}


data_lists_file <- "~/shoe_nnet/GPU_training/noaug_data_lists.rdata"


if(!file.exists(data_lists_file)) {
  base_dir <- ifelse(gpu, "/work/CSAFE/shoes/onehot",
                     "~/shoe_nnet/shoes/onehot")
  train_dir <- file.path(base_dir, "train")
  # train_dir <- file.path(base_dir, "train_aug")
  validation_dir <- file.path(base_dir, "validation")
  test_dir <- file.path(base_dir, "test")
  
  get_labs <- function(directory, verbose = F) {
    
    sample_count <- length(list.files(directory))
    labels <- array(0, dim = c(sample_count, length(classes)))
    files <- list.files(directory)
    
    for (i in 1:sample_count) {
      if (verbose) cat(paste(i, ", ", sep=""))
      
      fname <- files[i]
      str <- substr(fname, 1, regexpr("-",fname)-1)
      for (j in 1:length(classes)) {
        labels[i, j] <- grepl(classes[j], str)
      }
    }
    labels
  }
  
  dir_to_data <- function(dir, verbose = F) {
    files <- list.files(dir, full.names = T)
    cat("Directory length:", length(files), "\n")
    
    if (verbose) {cat("Getting labels...\n")}
    labels <- get_labs(dir)
    
    if (verbose) {cat("Gathering data...\n")}
    data <- array(0, dim = c(length(files), 256, 256, 3))
    
    for (i in 1:length(files)) {
      if (verbose) {cat(i, ", ", sep = "")}
      img <- imager::load.image(files[i])
      dim(img) <- c(1, 256, 256, 3)
      data[i,,,] <- img
    }
    
    data <- data/255
    
    list(
      data = data, 
      labels = labels
    )
  }
  
  train <- dir_to_data(train_dir, verbose = T)
  validation <- dir_to_data(validation_dir, verbose = T)
  test <- dir_to_data(test_dir, verbose = T)
  
  save(train, validation, test, file = data_lists_file)

  } else {
    load(data_lists_file)
}

train$data <- train$data[(1:5000)*2,,,]
train$labels <- train$labels[(1:5000)*2,,,]


conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256, 256, 3)
)

model <- keras_model_sequential() %>% 
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu", 
              input_shape = 8 * 8 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = length(classes), activation = "sigmoid")

freeze_weights(conv_base)

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$data, train$labels,
  batch_size = 20,
  epochs = 3,
  validation_data = list(validation$data, validation$labels),
  shuffle = TRUE
)





