#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinyWidgets)
library(gridExtra)
library(jpeg)
library(grid)

# Define UI for application that draws a histogram
# ui <- fluidPage(
#   
#   # Application title
#   titlePanel("ggHeatmaps (CoNNOR Spatial Integration"),
#   sidebarLayout(
#     sidebarPanel(
#       )),
#   
#   # Sidebar with a slider input for number of bins 
#   mainPanel(
#     imageOutput("myImage")
#   )
# )


ui <- pageWithSidebar(
  headerPanel("ggHeatmaps (CoNNOR Spatial Integration)"),
  sidebarPanel(
    selectInput("image", "Choose an image",
                choices = list.files("www/."))
  ),
  mainPanel(
    # Use imageOutput to place the image on the page
    imageOutput("myImage")
  )
)

server <- function(input, output, session) {
  
  output$myImage <- renderImage({
    filename <- file.path("www", input$image)
    # Return a list containing the filename
    list(src = filename,
         contentType = 'image/jpg',
         width = 800,
         height = 500,
         alt = "This is alternate text")
  }, deleteFile = FALSE)
}


shinyApp(ui = ui, server = server)
