library(keras)
library(magrittr)
#install_keras()
use_backend("tensorflow")

gpu <- F

base_dir <- ifelse(gpu, "/work/CSAFE/shoes/circtriquad", "~/shoe_nnet/shoes/circtriquad")
test_dir <- file.path(base_dir, "test")

classes <- list.files(test_dir)

model <- load_model_hdf5("~/shoe_nnet/shoe_models/CircTriQuad/073118_vgg16_circtriquad_256_4.h5")

test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_dir, 
  test_datagen, 
  target_size = c(256, 256),
  batch_size = 1, 
  class_mode = "categorical",
  shuffle = F
)

test_files <- list.files(test_dir, recursive = T, full.names = T)

preds <- model %>%
  predict_generator(test_generator, steps = length(test_files), verbose = 1)

preds.df <- data.frame(round(preds, 2))
names(preds.df) <- classes

preds.df$pred <- classes[apply(preds, 1, which.max)]
preds.df$truth <- rep(classes, each = length(test_files)/length(classes))
preds.df$files <- test_files

table(truth = preds.df$truth, pred = preds.df$pred)

wrong <- preds.df[preds.df$pred != preds.df$truth,]

library(imager)

### plot all wrong images
small = .01
big = .6
par(mfrow = c(4,8), mar = c(small, small, big, small))
for (i in 1:nrow(wrong)) {
  plot(load.image(wrong[i, "files"]), axes = F)
  
}

### plot wrong images by truth category
for (i in classes) {
  wrong.c <- wrong[wrong$truth == i,]
  n <- nrow(wrong.c)
  nrow <- 2
  par(mfrow = c(nrow, ceiling(n/nrow)))

  for (j in 1:n) {
    plot(load.image(wrong.c[j, "files"]), axes = F, 
         main = paste("truth = ", substr(i, 1, 3),
                      ", label = ", substr(wrong.c[j, "pred"], 1, 3),
                      sep = ""), 
         cex.main = .7)
  }
}
