library(shiny)
library(dplyr)
library(data.table)
library(DT)
library(magrittr)


mod.num <- 3

choose.model <- function(n) {
  if (n==1) { #first onehot model, before quality control
    mf <- "~/shoe_nnet/shoe_models/OneHot/081518_vgg16_onehot_256_2.Rdata"
    imfolder <- "onehotV1"
    }
  if (n==2) { #after quality control, 3 classes
    mf = "~/shoe_nnet/shoe_models/OneHot/090818_vgg16_onehot_3class_256_2.Rdata"
    imfolder <- "onehotV2"
    }
  if (n==3) { #after QC, 8 classes
    mf = "~/shoe_nnet/shoe_models/OneHot/090918_vgg16_onehot_8class_256_2.Rdata"
    imfolder <- "onehotV2"
    }
  return(list(model.file = mf, folder = imfolder))
}
mod.list <- choose.model(mod.num)
model.file <- mod.list$model.file
folder <- mod.list$folder

ui <- fluidPage(
  
  # Application title
  titlePanel("Neural Network Shoeprint Shape Predictions"),
  
  mainPanel(
    DT::dataTableOutput("out")
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  load(model.file)
  classes <- colnames(preds)
  fnames <- list.files(paste("~/shoe_nnet/nnet_pred_viewer/www/", folder, sep=""))
  truth <- test_labs
  colnames(truth) <- colnames(test_labs) %>% 
    paste("truth", ., sep="_")
  
  html_fnames <- sprintf('<img src ="%s" width = "100%%"/>', 
                         paste(folder, fnames, sep="/"))
  
  obj <- data.table(cbind(Image = html_fnames, round(preds, 2), truth))
  set.seed(2)
  obj <- obj[sample(1:length(fnames), length(fnames)),]
  
  output$out <- DT::renderDataTable({
    datatable(obj,
              escape = F,
              extensions = c("FixedHeader"),
              options = list(
                fixedHeader = T,
                pageLength = 100,
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

