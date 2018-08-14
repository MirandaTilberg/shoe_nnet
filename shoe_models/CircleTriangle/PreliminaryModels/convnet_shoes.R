# original_dataset_dir_circle <- "~/shoes/original data/circle"
# original_dataset_dir_triangle <- "~/shoes/original data/triangle"
base_dir <- "~/shoes/circletriangle"
# dir.create(base_dir)
# 
# 
train_dir <- file.path(base_dir, "train")
# dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
# dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
# dir.create(test_dir)
# 
train_circle_dir <- file.path(train_dir, "circle")
# dir.create(train_circle_dir)
# 
train_triangle_dir <- file.path(train_dir, "triangle")
# dir.create(train_triangle_dir)
# 
validation_circle_dir <- file.path(validation_dir, "circle")
# dir.create(validation_circle_dir)
# 
validation_triangle_dir <- file.path(validation_dir, "triangle")
# dir.create(validation_triangle_dir)
# 
test_circle_dir <- file.path(test_dir, "circle")
# dir.create(test_circle_dir)
# 
test_triangle_dir <- file.path(test_dir, "triangle")
# dir.create(test_triangle_dir)
# 
# 
# circle_data <- list.files(path = "shoes/original data/circle/")
# triangle_data <- list.files(path = "shoes/original data/triangle")
# set.seed(1)
# 
# cs <- sample(x = circle_data, replace = F)
# ts <- sample(x = triangle_data, replace = F)
# 
# ##Circles
# 
# fnames <- cs[1:(trunc(length(cs)/2))]
# file.copy(file.path(original_dataset_dir_circle, fnames), 
#           file.path(train_circle_dir))
# 
# fnames <- cs[(trunc(length(cs)/2)+1):(trunc(length(cs)*3/4))]
# file.copy(file.path(original_dataset_dir_circle, fnames),
#           file.path(validation_circle_dir))
# 
# fnames <- cs[(trunc(length(cs)*3/4)+1):length(cs)]
# file.copy(file.path(original_dataset_dir_circle, fnames),
#           file.path(test_circle_dir))
# 
# ##Triangles
# 
# fnames <- ts[1:(trunc(length(ts)/2))]
# file.copy(file.path(original_dataset_dir_triangle, fnames), 
#           file.path(train_triangle_dir))
# 
# fnames <- ts[(trunc(length(ts)/2)+1):(trunc(length(ts)*3/4))]
# file.copy(file.path(original_dataset_dir_triangle, fnames),
#           file.path(validation_triangle_dir))
# 
# fnames <- ts[(trunc(length(ts)*3/4)+1):length(ts)]
# file.copy(file.path(original_dataset_dir_triangle, fnames),
#           file.path(test_triangle_dir))
# 
# cat("total training circle images:", length(list.files(train_circle_dir)), "\n")
# cat("total training triangle images:", length(list.files(train_triangle_dir)), "\n")
# cat("total validation circle images:", length(list.files(validation_circle_dir)), "\n")
# cat("total validation triangle images:", length(list.files(validation_triangle_dir)), "\n")
# cat("total test circle images:", length(list.files(test_circle_dir)), "\n")
# cat("total test triangle images:", length(list.files(test_triangle_dir)), "\n")


library(magrittr)
library(keras)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(256, 256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)


# All images will be rescaled by 1/255
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  # This is the target directory
  train_dir,
  # This is the data generator
  train_datagen,
  # All images will be resized to 150x150
  target_size = c(256,256),
  batch_size = 20,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(256, 256),
  batch_size = 20,
  class_mode = "binary"
)

batch <- generator_next(train_generator)
str(batch)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("circle_and_triangle_small_1.h5")

plot(history)

########################################################################

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# We pick one image to "augment"
fnames <- list.files(train_circle_dir, full.names = TRUE)
img_path <- fnames[[2]]

# Convert it to an array with shape (150, 150, 3)
img <- image_load(img_path, target_size = c(256, 256))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 256, 256, 3))

# Generated that will flow augmented images
augmentation_generator <- flow_images_from_data(
  img_array, 
  generator = datagen, 
  batch_size = 1 
)

# Plot the first 4 augmented images
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)


model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(256,256, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")  

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(256, 256),
  batch_size = 32,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(256, 256),
  batch_size = 32,
  class_mode = "binary"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

plot(history)

model %>% save_model_hdf5("circle_and_triangle_small_2.h5")
