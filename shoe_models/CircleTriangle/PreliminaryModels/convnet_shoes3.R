library(keras)

model <- load_model_hdf5("circle_and_triangle_small_2.h5")
summary(model)  # As a reminder.

img_path <- "~/shoes/circletriangle/test/triangle/aerosoles-pin-down-black-leather_product_9010211_color_72_crop2.jpg"

# We preprocess the image into a 4D tensor
img <- image_load(img_path, target_size = c(256, 256))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 256, 256, 3))

# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor <- img_tensor / 255

dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))


# Extracts the outputs of the top 8 layers:
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

# Returns a list of five arrays: one array per layer activation
activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)


plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, 
        col = gray((1:12)/12))
}

plot_channel(first_layer_activation[1,,,5])

plot_channel(first_layer_activation[1,,,7])

dir.create("triangle_activations")
image_size <- 58
images_per_row <- 16

for (i in 1:8) {
  
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("triangle_activations/", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}

