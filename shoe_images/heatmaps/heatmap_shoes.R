library(keras)
library(imager)
library(tidyr)
library(magick) 
library(viridis)
source("/home/tiltonm/shoe_nnet/helper_scripts/get_most_recent.R")


### Set script parameters (choose model/image/directories, add labels?) -----

# Where the images will come from
image_dir <- "/home/tiltonm/shoe_nnet/shoes/onehot/test"

# Choose image index (use 0 to randomly select image from image_dir)
specify_index <- 5000

# File containing weights from model
model_wts_file <- get_most_recent("weights", verbose = T)

# Should labels be attached to saved heatmaps?
fixed_labels <- T

# Where to create the heatmap working directory
dir_path <- "/home/tiltonm/shoe_nnet/shoe_images/heatmaps/shoes"

# Choose file to fill in failed heatmaps
fail_file <- file.path(dir_path, "failure_img.png")


### Load weights into model structure --------------------------------------

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


### Set-up working directory for heatmap process ---------------------------

name <- "shoes-"
index <- ifelse(specify_index,
                specify_index,
                sample(1:length(list.files(image_dir)), 1))
prefix <- paste0(name,as.character(index), "_")

path <- file.path(dir_path, prefix)
dir.create(path)

img_path <- list.files(image_dir, full.names = T)[index]
file.copy(img_path, path)


### Heatmap helper functions -----------------------------------------------

plot_heatmap <- function(heatmap, width = 256, height = 256,
                         bg = "white", col = terrain.colors(12)) {
  op = par(mar = c(0,0,0,0))
  on.exit({par(op)}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

write_heatmap <- function(heatmap, filename, width = 256, height = 256,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}


### Use trained model to make heatmap arrays for each class ----------------

classes <- c("bowtie", "chevron", "circle", "line", 
             "polygon", "quad", "star", "text", "triangle")
n_classes <- length(classes)

img <- load.image(img_path)
dim(img) <- c(1, 256, 256, 3)

predictions <- model %>% 
  predict(img) %>% 
  as.vector() %>% 
  set_names(classes)

true_labels <- sapply(classes, function(x){grepl(x, basename(img_path))}) %>%
  as.numeric()

heatmap <- array(dim = c(n_classes, 16, 16))
successful_heatmap <- c()

for (j in 1:n_classes) {
  # Make 16x16 heatmap matrix for each class
  img_output <- model$output[,j]
  
  last_conv_layer <- model %>% get_layer("block5_conv3")
  grads <- k_gradients(img_output, last_conv_layer$output)[[1]]
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
  
  # Check if the heatmap matrix contains any NaN values
  if (!anyNA(heatmap[j,,])) {
    successful_heatmap <- c(successful_heatmap, j)
  } 
}

# Turn successful heatmap arrays into overlays and apply to original image
### (If no array was successful, delete working directory for this image)

if (is.null(successful_heatmap)) {
  
  unlink(path, recursive = T)
  cat("No heatmaps were successful\n")
  
} else{
  
  cat("Successful heatmaps:", successful_heatmap)
  
  image <- image_read(img_path)
  info <- image_info(image) 
  geometry <- sprintf("%dx%d!", info$width, info$height) 
  
  pal <- col2rgb(viridis(20), alpha = TRUE) 
  alpha <- floor(seq(0, 255, length = ncol(pal))) 
  pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
  
  for (j in 1:n_classes){
    
    # Generate label files (e.g. "Class: prob")
    label <- paste0(classes[j],
                    "(", true_labels[j], "): ",
                    round(predictions[j],3))
    
    labels_file <- file.path(path, paste0(prefix, j, "labels", ".png"))
    
    png(labels_file, width = 256, height = 50)
    par(mar = c(0,0,0,0))
    plot(0:1, 0:1, ann = F, bty = 'n', type = 'n', xaxt = 'n', yaxt = 'n')
    text(x = .5, y = .5, cex = 2.1, col = "black", label)
    dev.off()
    
    # Create overlay files and apply them to the original image
    if(j %in% successful_heatmap){
      
      heatmap_file <- file.path(path, paste0(prefix, j, "heatmap", ".png"))
      write_heatmap(heatmap, heatmap_file)
      
      overlay_file <- file.path(path, paste0(prefix, j, "overlay", ".png"))
      write_heatmap(heatmap[j,,], overlay_file, 
                    width = 14, height = 14, bg = NA, col = pal_col) 
      
      overlaid_file <- file.path(path, paste0(prefix, j, "overlaid", ".png"))
      image_read(overlay_file) %>%
        image_resize(geometry, filter = "quadratic") %>% 
        image_composite(image, operator = "blend", compose_args = "20") %>%
        image_write(., overlaid_file)
      
    } else {
      overlaid_file <- fail_file
    }
    
    full_file <- file.path(path, 
                           paste0(prefix, classes[j],
                                  "_labeled_heatmap", ".png"))
    
    # append labels to overlaid heatmap images if desired
    if (fixed_labels) {
      final <- imlist(load.image(overlaid_file), 
                      load.image(labels_file)) %>%
        imappend(., axis = "y")
    } else {
      final <- load.image(overlaid_file)
    }
    
    # plot and save each image individually
    par(mar = rep(0,4), mfrow = c(1,1))
    plot(final, axes = F)
    dev.copy(png, full_file)
    dev.off()
    
    # clean out intermediate files
    if(j %in% successful_heatmap) {
      file.remove(overlaid_file, labels_file, overlay_file, heatmap_file)
    } else {
      file.remove(labels_file)
    }
    
  }
}

# plot all images in one grid and save as one full file
par(mfrow = c(3,4), mar = c(0,0,0,0))
plot(imager::load.image(img_path), axes = F)

for (j in 1:n_classes) {
  full_file <- file.path(path, paste0(prefix, classes[j],
                                      "_labeled_heatmap", ".png"))
  plot(imager::load.image(full_file), axes = F)
}

dev.copy(png, file.path(path, paste0("final_", prefix, 
                                     "labeled_heatmap", ".png")))
dev.off()
dev.off()
index
