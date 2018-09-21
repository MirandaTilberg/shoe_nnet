library(magrittr)
library(keras)
use_backend("tensorflow")
#install_keras()

gpu <- F

base_dir <- ifelse(gpu, "/work/CSAFE/shoes/onehot", "~/shoe_nnet/shoes/onehot")
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
train_aug_dir <- file.path(base_dir, "train_aug")
dir.create(train_aug_dir)

n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))



#dim(save_train$features)
library(stringr)
fnames <- list.files(train_dir)
fnames.front <- str_remove(fnames, ".jpg")

library(jpeg)

for (i in 1:n_train) {
  cat(paste(i, ", ", sep = ""))
  img.loc <- file.path(train_dir, fnames[i])
  file.copy(from = img.loc, to = train_aug_dir)
  
  
  img <- readJPEG(img.loc)
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
