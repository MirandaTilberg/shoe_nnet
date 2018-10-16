library(keras)
library(magrittr)
library(imager)
library(tidyr)
library(magick) 
library(viridis)

# Set parameters for saving files
path <- "/home/tiltonm/shoe_nnet/shoe_images/heatmaps"
folder <- "onehot-"
index <- sample(1:5659, 1)
prefix <- paste0(folder,as.character(index), "_")

# Choose image and extract model predictions
model <- application_vgg16(weights = "imagenet")
img_path <- list.files("shoes/onehot/test/", full.names = T)[index] #1:5659
img <- image_load(img_path, target_size = c(224, 224)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 224, 224, 3)) %>%
  imagenet_preprocess_input()


preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]

# Format predictions and save as label image
preds_df <- imagenet_decode_predictions(preds, top = 3)[[1]][,2:3]
names(preds_df) <- c("class", "p")
preds_df$p <- round(preds_df$p, 3)
labels <- unite(preds_df, sep = ": ")[,1,drop=T]

labels_file <- file.path(path, paste0(prefix, "labels", ".png"))

png(labels_file, width = 2*224, height = 2*100)
plot(c(0, 1), c(.2,.85), ann = F, bty = 'n', type = 'n', xaxt = 'n', yaxt = 'n')
text(x = 0.45, y = .72, cex = 1.8, col = "black", labels[1])
text(x = 0.45, y = .5, cex = 1.8, col = "black", labels[2])
text(x = 0.45, y = .28, cex = 1.8, col = "black", labels[3])
dev.off()

label_img <- crop.borders(load.image(labels_file), nx = 96, ny = 100/2)
imager::save.image(im = label_img, file = labels_file)

# Create heatmap image
img_output <- model$output[,which.max(preds)]
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

heatmap_file <- file.path(path, paste0(prefix, "heatmap", ".png"))
write_heatmap(heatmap, heatmap_file) 


# Read the original image and it's geometry
image <- image_read(img_path)
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

# Create a blended / transparent version of the heatmap image
pal <- col2rgb(viridis(20), alpha = TRUE) 
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)

overlay_file <- file.path(path, paste0(prefix, "overlay", ".png"))
write_heatmap(heatmap, overlay_file, 
              width = 14, height = 14, bg = NA, col = pal_col) 


# Overlay the heatmap and save 
overlaid_file <- file.path(path, paste0(prefix, "overlaid", ".png"))
image_read(overlay_file) %>%
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  image_write(., overlaid_file)




full_file <- file.path(path, paste0(prefix, "labeled_heatmap", ".png"))
combo <- imlist(load.image(overlaid_file), load.image(labels_file))

final <- imappend(combo, axis = "y")
par(mar = rep(0,4))
plot(final, axes = F)
dev.copy(png, full_file)
dev.off()



# png(full_file, height = 356, width = 256)
# 
# dev.off()


#imager::save.image(im = final, file = full_file)

file.remove(overlaid_file, labels_file, overlay_file, heatmap_file)



