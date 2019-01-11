library(magrittr)
library(ggplot2)
library(RColorBrewer)
source("/home/tiltonm/shoe_nnet/helper_scripts/get_most_recent.R")

########## Choose model file and detect classes
predictions_file <- get_most_recent(verbose = T)
load(predictions_file)

classes <- colnames(preds)
n_per_class <- apply(test_labs, 2, sum)
n_per_class

n_classes <- length(classes)
preds <- preds %>% round(2)


########## Create confusion matrix
confusion_mat <- matrix(0, ncol = n_classes, nrow = n_classes)
class_rel <- matrix(0, ncol = n_classes, nrow = n_classes)

for (i in 1:n_classes) {
  true_i <- test_labs[,i]==1
  n_i <- sum(true_i)
  
  class_rel[i,] <- test_labs[true_i,] %>% apply(., 2, sum)
  
  preds_i <- preds
  preds_i[,-i] <- (preds_i[,-i] - test_labs[,-i])
  preds_i[preds_i < 0] <- NA
  
  
  f <- function(vec) {
    thresh = .2
    sum(vec > thresh, na.rm = T)
  }
  
  
  #metric <- apply(preds_i[true_i,], 2, max, na.rm=T) %>% round(2)
  #metric <- apply(preds_i[true_i,], 2, mean, na.rm=T) %>% round(2)
  metric <- (apply(preds_i[true_i,], 2, f) / n_i) %>% round(2)
  
  confusion_mat[i,] <- metric
}

colnames(confusion_mat) <- rownames(confusion_mat) <- classes
colnames(class_rel) <- rownames(class_rel) <- classes

########## Plot correlation matrix
source("/home/tiltonm/shoe_nnet/shoe_images/confusion_matrices/ggcorrplot2.R")
cols <- colorRampPalette(c("white", "cornflowerblue"))
conf_plot <- ggcorrplot2(confusion_mat, hc.order = F, outline.col = "white", 
                         colors = c("white", cols(2)), legend.title = "Value",
                         lab = T, lab_size = 3.5, show.legend = F) + 
  xlab("Truth") + ylab("Predictions") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))

cols <- colorRampPalette(c("white", "plum"))
quantity_mat_plot <- ggcorrplot2(class_rel, limit = c(0, max(class_rel)),
                                 colors = c("white", cols(2)), 
                                 legend.title = "Quantity", lab = T,
                                 show.legend = F) +
  theme(axis.title.x = ggplot2::element_blank(),
        axis.title.y = ggplot2::element_blank())


quantity.df <- data.frame(classes = stringr::str_to_title(classes), 
                          single_class = diag(class_rel) , 
                          multi_class = (class_rel - diag(diag(class_rel))) %>%
                            apply(., 1, sum)) %>%
  reshape2::melt(., id.vars = "classes")
quantity.df$variable <- factor(quantity.df$variable,
                               levels = c("multi_class", "single_class"), 
                               labels = c("Multi-Class", "Single Class"))

quantity_bar_plot <- ggplot(quantity.df, aes(x = classes, y = value,
                                             fill = variable)) +
  geom_bar(stat='identity', color = "black") +
  xlab("Classes") +
  ylab("Quantity") +
  labs(fill='Image Type') +
  scale_fill_manual(values=c("#A6CEE3", "#1F78B4")) +
  theme_minimal(base_size = 15) +
  theme(panel.grid.major.y = element_blank()) +
  scale_x_discrete(limits = rev(levels(quantity.df$classes))) +
  coord_flip() + 
  theme(legend.margin=margin(t = 0, unit='cm'))

conf_plot
ggsave(filename = file.path("~/shoe_nnet/shoe_images/confusion_matrices",
                            paste0("ConfMat_", names(predictions_file),
                                   ".png")),
       height = 5.5, width = 5.5)

quantity_mat_plot

quantity_bar_plot
ggsave(filename = file.path("~/shoe_nnet/shoe_images/confusion_matrices",
                            paste0("QuantityBars_", names(predictions_file),
                                   ".png")),
       height = 4/1.5, width = 10/1.5)
