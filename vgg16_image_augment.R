library(magrittr)
library(stringr)
library(jpeg)
library(keras)
use_backend("tensorflow")
#install_keras()

gpu <- F
view <- T

base_dir <- ifelse(gpu, "/work/CSAFE/shoes/onehot", "~/shoe_nnet/shoes/onehot")
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

train_aug_dir <- file.path(base_dir, "train_aug_exp7")
suppressWarnings(dir.create(train_aug_dir))

n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))


fnames <- list.files(train_dir)
fnames.front <- str_remove(fnames, ".jpg")

if (length(list.files(train_aug_dir, pattern = "aug_")) < 2000) {
  #for (i in 1:n_train) {
  vec <- sample(1:n_train, 50)
  for (i in vec) {
    cat(paste(i, ", ", sep = ""))
    img.loc <- file.path(train_dir, fnames[i])
    file.copy(from = img.loc, to = train_aug_dir)
    
    
    img <- readJPEG(img.loc)
    dim(img) <- c(1, dim(img))
    
    
    gen_images <- image_data_generator(rotation_range = 40,
                                       width_shift_range = 0.05,
                                       height_shift_range = 0.05,
                                       shear_range = 60,
                                       zoom_range = 0.1,
                                       channel_shift_range = .1,
                                       zca_whitening = T,
                                       samplewise_std_normalization = T,
                                       vertical_flip = T,
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
}

if(view) {
  library(imager)
  
  choice <- sample(vec,1)#8083 #sample(1:11318, 1)
  aug_files <- list.files(train_aug_dir)
  aug_files_full <- list.files(train_aug_dir, full.names = T)

  im_nums <- fnames.front[choice] %>% 
    grep(., aug_files, fixed = T) %>% 
    rev
  

  par(mfrow = c(2,2), mar = rep(.1, 4))
  for (i in 1:4) {
    img <- imager::load.image(aug_files_full[im_nums[i]])
    plot(img, axes = F)
  }
}

