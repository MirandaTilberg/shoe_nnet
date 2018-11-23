library(keras)
model_base <- c("resnet50", "vgg16")[2]


if (model_base == "resnet50") {
  model <- application_resnet50(weights = "imagenet", 
                                include_top = FALSE)
} else {
  model <- application_vgg16(weights = "imagenet", 
                             include_top = FALSE)
}

layers <- capture.output(summary(model), file = NULL)
layers <- layers[grepl("Conv2D", layers)] %>%
  gsub('([A-Za-z0-9-]+) .*', '\\1', .)

layers
layer_name <- layers[5]
filter_index <- 1

layer_output <- get_layer(model, layer_name)$output
loss <- k_mean(layer_output[,,,filter_index])

# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads <- k_gradients(loss, model$input)[[1]] 

# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)

iterate <- k_function(list(model$input), list(loss, grads))

# Let's test it
c(loss_value, grads_value) %<-%
  iterate(list(array(0, dim = c(1, 150, 150, 3))))

# We start from a gray image with some noise
input_img_data <-
  array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128 

step <- 1  # this is the magnitude of each gradient update
for (i in 1:40) { 
  # Compute the loss value and gradient value
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  # Here we adjust the input image in the direction that maximizes the loss
  input_img_data <- input_img_data + (grads_value * step)
}

deprocess_image <- function(x) {
  
  dms <- dim(x)
  
  # normalize tensor: center on 0., ensure std is 0.1
  x <- x - mean(x) 
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1 
  
  # clip to [0, 1]
  x <- x + 0.5 
  x <- pmax(0, pmin(x, 1))
  
  # Reshape to original image dimensions
  array(x, dim = dms)
}

generate_pattern <- function(layer_name, filter_index, size = 150) {
  
  # Build a loss function that maximizes the activation
  # of the nth filter of the layer considered.
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index]) 
  
  # Compute the gradient of the input picture wrt this loss
  grads <- k_gradients(loss, model$input)[[1]]
  
  # Normalization trick: we normalize the gradient
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  # This function returns the loss and grads given the input picture
  iterate <- k_function(list(model$input), list(loss, grads))
  
  # We start from a gray image with some noise
  input_img_data <- 
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  # Run gradient ascent for 40 steps
  step <- 1
  for (i in 1:40) {
    cat(i, ", ", sep = "")
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step) 
  }
  cat("\n")
  img <- input_img_data[1,,,]
  deprocess_image(img) 
}

library(grid)
par(mfrow = c(1,1))
#grid.raster(generate_pattern(layer_name, 1))
grid.raster(generate_pattern("block5_conv3", 41))

library(grid)
library(gridExtra)
dir <- file.path("/home/tiltonm/shoe_nnet/shoe_images/filters",
                 paste(model_base, "filters", sep = "_"))
dir.create(dir)
for (layer_name in sample(layers)) {
  img_file <- paste0(dir, "/", layer_name, ".png")
  if(!file.exists(img_file)) {
    cat(layer_name, ":\n", sep="") 
    size <- 140
    
    grid_size <- c(8,8)
    png(img_file, width = grid_size[1] * size, height = grid_size[2] * size)
    
    grobs <- list()
    for (i in 0:(grid_size[1]-1)) {
      for (j in 0:(grid_size[2]-1)) {
        cat(i + (j*grid_size[1]) + 1, ", ", sep = "")
        pattern <- generate_pattern(layer_name, i + (j*grid_size[1]) + 1, size = size)
        grob <- rasterGrob(pattern, 
                           width = unit(0.9, "npc"), 
                           height = unit(0.9, "npc"))
        grobs[[length(grobs)+1]] <- grob
      }  
      cat("\n")
    }
    cat("\n")
    grid.arrange(grobs = grobs, ncol = grid_size[2])
    dev.off()
  }
}
