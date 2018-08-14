library(keras)

model <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE
)

layer_name <- "block3_conv1"
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
c(loss_value, grads_value) %<-% iterate(list(input_img_data))
input_img_data <- input_img_data + (grads_value * step) 
}

img <- input_img_data[1,,,]
deprocess_image(img) 
}

library(grid)
grid.raster(generate_pattern("block3_conv1", 1))

library(grid)
library(gridExtra)
dir.create("vgg_filters")
for (layer_name in c("block1_conv1", "block2_conv1", 
                     "block3_conv1", "block4_conv1")) {
  size <- 140
  
  png(paste0("vgg_filters/", layer_name, ".png"),
      width = 8 * size, height = 8 * size)
  
  grobs <- list()
  for (i in 0:7) {
    for (j in 0:7) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern, 
                         width = unit(0.9, "npc"), 
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }  
  }
  
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
}

# Clear out the session
#k_clear_session()

# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model <- application_vgg16(weights = "imagenet")

# The local path to our target image
img_path <- "~/creative_commons_elephant.jpg"

# Start witih image of size 224 by 224
img <- image_load(img_path, target_size = c(224, 224)) %>% 
  # Array of shape (224, 224, 3)
  image_to_array() %>% 
  # Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  # Preprocesses the batch (this does channel-wise color normalization)
  imagenet_preprocess_input()

preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]

which.max(preds[1,])

# This is the "african elephant" entry in the prediction vector
african_elephant_output <- model$output[, 387]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer <- model %>% get_layer("block5_conv3")

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads <- k_gradients(african_elephant_output, last_conv_layer$output)[[1]]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,])) 

# These are the values of these two quantities, as arrays,
# given our sample image of two elephants
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for (i in 1:512) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)

write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

write_heatmap(heatmap, "images/elephant_heatmap.png")


library(magick) 
library(viridis) 

# Read the original elephant image and it's geometry
image <- image_read(img_path)
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

# Create a blended / transparent version of the heatmap image
pal <- col2rgb(viridis(20), alpha = TRUE) 
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "images/elephant_overlay.png", 
              width = 14, height = 14, bg = NA, col = pal_col) 

# Overlay the heatmap
image_read("images/elephant_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot() 
