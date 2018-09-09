library(shiny)
library(dplyr)
library(data.table)
library(DT)
library(magrittr)

folder <- "onehotV2"
ui <- fluidPage(
  
  # Application title
  titlePanel("Neural Network Shoeprint Shape Predictions"),
  
  mainPanel(
    DT::dataTableOutput("test")
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  # load("~/shoe_nnet/shoe_models/OneHot/081518_vgg16_onehot_256_2.Rdata")
  load("~/shoe_nnet/shoe_models/OneHot/090818_vgg16_onehot_3class_256_2.Rdata")
  classes <- colnames(preds)
  fnames <- list.files("~/shoe_nnet/nnet_pred_viewer/www/onehotV2")
  truth <- test_labs
  colnames(truth) <- colnames(test_labs) %>% 
    paste("truth", ., sep="_")
  
  # thresh <- 0.4
  # 
  # merged <- truth
  # 
  # for (i in 1:length(classes)) {
  #   if
  # }
  #   
  
  html_fnames <- sprintf('<img src ="%s" width = "100%%"/>', 
                         #paste("onehotV1/", fnames, sep=""))
                         paste("onehotV2/", fnames, sep=""))
  
  obj <- data.table(cbind(Image = html_fnames, round(preds, 2), truth))
  set.seed(1)
  obj <- obj[sample(1:length(fnames), length(fnames)),]
  
  output$test <- DT::renderDataTable({
    datatable(obj,
              escape = F,
              extensions = c("FixedHeader"),
              options = list(
                fixedHeader = T,
                columnDefs = list(list(targets = 
                                         (ncol(obj)-length(classes)+1):(ncol(obj)), 
                                       visible = FALSE))
              )) %>%
      formatStyle(classes, 
                  valueColumns = paste("truth", classes, sep="_"), 
                  target = "cell", 
                  backgroundColor = styleEqual(1, "palegreen")
      )
  })
}

# Run the application 
setwd("~/shoe_nnet/nnet_pred_viewer/")
shinyApp(ui = ui, server = server)

