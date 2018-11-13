library(keras)
library(imager)
library(tidyr)
library(magick) 
library(viridis)

# Load trained model and choose image ---------------------------------------

model_wts_file <- "~/shoe_nnet/test_wts.h5"

input <- layer_input(shape = c(256, 256, 3))

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_tensor = input)

output <- conv_base$output %>%
  layer_flatten(input_shape = input) %>%
  layer_dense(units = 256, activation = "relu",
              input_shape = 8 * 8 * 512) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 9, activation = "sigmoid")

model <- keras_model(input, output)

load_model_weights_hdf5(model, model_wts_file, by_name = T)


# Use trained model to make heatmaps for each class -------------------------

### Parameters for saving files

dir_path <- "/home/tiltonm/shoe_nnet/shoe_images/heatmaps"
image_dir <- "/home/tiltonm/shoe_nnet/shoes/onehot/test"

name <- "shoes-"
index <- sample(1:length(list.files(image_dir)), 1)
prefix <- paste0(name,as.character(index), "_")

path <- file.path(dir_path, prefix)
# if (file.exists(path)) {
#   prefix <- paste0(prefix, "(1)_")
#   path <- file.path(dir_path, prefix)
#   
# }
dir.create(path)

img_path <- list.files(image_dir, full.names = T)[index]
img <- load.image(img_path)
dim(img) <- c(1, 256, 256, 3)

file.copy(img_path, path)


# Heatmap helper functions
plot_heatmap <- function(heatmap, width = 224, height = 224,
                         bg = "white", col = terrain.colors(12)) {
  op = par(mar = c(0,0,0,0))
  on.exit({par(op)}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}


predictions <- model %>% predict(img)
#classes <- c("bowtie", "chevron", "circle", "hexagon", "line", 
#             "pentagon", "quad", "star", "text", "triangle")
classes <- c("bowtie", "chevron", "circle","line", 
             "polygon", "quad", "star", "text", "triangle")
n_classes <- length(classes)


heatmap <- array(dim = c(n_classes, 16, 16))
par(mfrow = c(3,4))
successful_heatmap <- c()

for (j in 1:n_classes) {
  
  img_output <- model$output[,j]
  
  last_conv_layer <- model %>% get_layer("block5_conv3")
  grads <- k_gradients(img_output, last_conv_layer$output)[[1]]
  
  # finish making the heatmap
  pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
  iterate <- k_function(list(model$input),
                        list(pooled_grads, last_conv_layer$output[1,,,]))
  c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
  for (i in 1:512) {
    conv_layer_output_value[,,i] <-
      conv_layer_output_value[,,i] * pooled_grads_value[[i]]
  }
  heatmap[j,,] <- apply(conv_layer_output_value, c(1,2), mean)
  heatmap[j,,] <- pmax(heatmap[j,,], 0)
  heatmap[j,,] <- heatmap[j,,] / max(heatmap[j,,])
  
  
  if (!anyNA(heatmap[j,,])) {
    successful_heatmap <- c(successful_heatmap, j)
    
  }
}

successful_heatmap

if (is.null(successful_heatmap)) {
  
  unlink(path, recursive = T)
  
} else{
  
image <- image_read(img_path)
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 


pal <- col2rgb(viridis(20), alpha = TRUE) 
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)

for (j in successful_heatmap) {
  plot_heatmap(heatmap[j,,], width = N, height = N)
  heatmap_file <- file.path(path, paste0(prefix, j, "heatmap", ".png"))
  write_heatmap(heatmap, heatmap_file)
  
  label <- paste0(classes[j], ": ", round(predictions[j],3))
  
  labels_file <- file.path(path, paste0(prefix, j, "labels", ".png"))
  
  png(labels_file, width = 256, height = 50)
  par(mar = c(0,0,0,0))
  plot(0:1, 0:1, ann = F, bty = 'n', type = 'n', xaxt = 'n', yaxt = 'n')
  text(x = .5, y = .5, cex = 2.1, col = "black", label)
  dev.off()
  
  overlay_file <- file.path(path, paste0(prefix, j, "overlay", ".png"))
  write_heatmap(heatmap[j,,], overlay_file, 
                width = 14, height = 14, bg = NA, col = pal_col) 
  
  overlaid_file <- file.path(path, paste0(prefix, j, "overlaid", ".png"))
  image_read(overlay_file) %>%
    image_resize(geometry, filter = "quadratic") %>% 
    image_composite(image, operator = "blend", compose_args = "20") %>%
    image_write(., overlaid_file)
  
  full_file <- file.path(path, paste0(prefix, j, "labeled_heatmap", ".png"))
  combo <- imlist(load.image(overlaid_file), load.image(labels_file))
  
  final <- imappend(combo, axis = "y")
  
  par(mar = rep(0,4), mfrow = c(1,1))
  plot(final, axes = F)
  dev.copy(png, full_file)
  dev.off()
  
  file.remove(overlaid_file, labels_file, overlay_file, heatmap_file)
}
}

par(mfrow = c(3,4), mar = c(0,0,0,0))
plot(imager::load.image(img_path), axes = F)

for (j in successful_heatmap) {
  full_file <- file.path(path, paste0(prefix, j, "labeled_heatmap", ".png"))
  plot(imager::load.image(full_file), axes = F)
}

dev.copy(png, file.path(path, paste0(prefix, "labeled_heatmap", ".png")))
dev.off()

