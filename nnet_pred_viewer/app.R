library(shiny)
library(dplyr)
folder <- "onehot"
ui <- fluidPage(
   
   # Application title
   titlePanel("Neural Network Shoeprint Shape Predictions"),
   
   # Sidebar with a slider input for number of bins 
   #sidebarLayout(
      #sidebarPanel(
         # sliderInput("nplot",
         #             "Number of images to display:",
         #             min = 1, max = 1, value = 1),
         # sliderInput("index",
         #             "Index of image to display:",
         #             min = 1, 
         #             max = length(list.files(paste("www/", folder, sep=""))), 
         #             value = 1)
      #),
      
      # Show a plot of the generated distribution
      mainPanel(
        #imageOutput("file"),
        #br(),
        DT::dataTableOutput("test")
        #DT::dataTableOutput("label")
      )
  # )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  
  load("~/shoe_nnet/shoe_models/OneHot/081518_vgg16_onehot_256_2.Rdata")
  classes <- colnames(preds)
  fnames <- list.files("~/shoe_nnet/nnet_pred_viewer/www/onehot")
  truth <- test_labs
  colnames(truth) <- colnames(test_labs) %>% 
    paste("truth", ., sep="_")
  
  output$file <- renderImage(fnames[input$index] %>%
                               paste("www/onehot/", ., sep="") %>%
                               list(src = ., width = "50%"),
                             deleteFile = F)
  
  sprintf('<img src ="%s" width = "20%%"/>', fnames[1:5])
  
  library(data.table)
  obj <- data.table({cbind(fnames, round(preds,2), truth)})
  obj
  
  output$test <- DT::renderDataTable(obj )

  output$label <- DT::renderDataTable({rbind(preds[input$index,], 
                                    test_labs[input$index,]) %>% 
      round(2) %>%
      set_rownames(c("pred", "truth")) %>%
      formatStyle(paste("truth", classes[1], sep = "_"),
                  )
      })

}

# Run the application 
setwd("~/shoe_nnet/nnet_pred_viewer/")
shinyApp(ui = ui, server = server)

