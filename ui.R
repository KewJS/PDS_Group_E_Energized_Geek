library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(randomForest)
library(Metrics)
library(imputeTS)
library(Amelia)
library(mlbench)
library(corrplot)
library(scales)
library(caTools)
library(tidyr)
library(rpart)
library(ROCR)
library(grid)
library(scales)
library(pROC)
library(caTools)
library(caret)
library(ggalluvial)

train_df <- read.csv('data/num_train_df.csv')
# impute_train_df <- rfImpute(h1n1_seasonal_vaccine ~ . ,
#                             train_df, 
#                             iter=2, 
#                             ntree=300)
impute_train_df <- na_mean(train_df)
impute_train_df <- impute_train_df[, order(names(impute_train_df))]

# Importing model
numeric_cols <- c()
for (i in colnames(impute_train_df)){
  if (class(impute_train_df[[i]]) != 'character'){
    numeric_cols <-  rbind(numeric_cols,i)
  }
}
numeric_cols <- as.vector(numeric_cols)

scaled_df <- impute_train_df%>%
  select(c(numeric_cols,c('h1n1_seasonal_vaccine')))%>%
  sapply(function(.) rescale(.))

scaled_df <- as.data.frame(scaled_df)%>%
  mutate_all(~ifelse(is.na(.), median(., na.rm=TRUE), .))

set.seed(101) 
sample = sample.split(scaled_df$h1n1_seasonal_vaccine, SplitRatio=0.70)
train = subset(scaled_df, sample==TRUE)
test  = subset(scaled_df, sample==FALSE)

model.h1 <- glm(h1n1_vaccine~.-seasonal_vaccine, family=binomial(link='logit'), data=train)
model.se <- glm(seasonal_vaccine~.-h1n1_vaccine, family=binomial(link='logit'), data=train)

h1n1.probs = predict(model.h1, type='response', newdata=test)
se.probs = predict(model.se, type='response', newdata=test)

# model_rf <- glm(h1n1_seasonal_vaccine~.-h1n1_seasonal_vaccine, family=binomial(link='logit'), data=train)
# y_pred = predict(model_rf, newdata = test)
# mae_rf = mae(test[[44]], y_pred)
# rmse_rf = rmse(test[[44]], y_pred)

####################################################################################################
# New Way of Training Random Forest
# intrain <- createDataPartition(y = impute_train_df$h1n1_seasonal_vaccine, p = 0.70, list = FALSE)
# train <- impute_train_df[intrain,]
# test <- impute_train_df[-intrain,]
set.seed(12345)
# model_rf <- randomForest(y= train[,43], train[,1:42],importance=T, confusion=T, err.rate=T)
model_rf <- readRDS(file = 'model/rf.rda')
y_pred <- predict(model_rf, test, type = "class")
rf.importancePlot <- varImpPlot(model_rf, sort=T, main="Feature Importance Plot for Vaccination (Seasonal & H1N1) Prediction")
####################################################################################################


before_missing.values <- train_df %>%
  gather(key = "key", value = "val") %>%
  mutate(isna = is.na(val)) %>%
  group_by(key) %>%
  mutate(total = n()) %>%
  group_by(key, total, isna) %>%
  summarise(num.isna = n()) %>%
  mutate(pct = num.isna / total * 100)
before_levels <- (before_missing.values  %>% filter(isna == T) %>%     
             arrange(desc(pct)))$key


levels <- (before_missing.values  %>% filter(isna == T) %>%     
             arrange(desc(pct)))$key

after_missing.values <- impute_train_df %>%
  gather(key = "key", value = "val") %>%
  mutate(isna = is.na(val)) %>%
  group_by(key) %>%
  mutate(total = n()) %>%
  group_by(key, total, isna) %>%
  summarise(num.isna = n()) %>%
  mutate(pct = num.isna / total * 100)
after_levels <- (after_missing.values  %>% filter(isna == T) %>%     
             arrange(desc(pct)))$key

input_cols <- c('h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask',
                'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face',
                'doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition', 'child_under_6_months', 'health_worker',
                'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'household_adults',
                'household_children', 'age_group', 'education', 'race', 'sex', 'income_poverty',
                'marital_status', 'rent_or_own', 'employment_status')

#R Shiny ui
ui <- dashboardPage(
  
  #Dashboard title
  dashboardHeader(title = 'SEASONAL & H1N1 FLU VACCINE', titleWidth = 290),
  
  #Sidebar layout
  dashboardSidebar(width = 290,
                   sidebarMenu(menuItem("Overview", tabName = "Overview", icon = icon('poll')),
                               menuItem("Data Summary", tabName = "Summary", icon = icon('cog')),
                               menuItem("Visualization", tabName = "Visualization", icon = icon('tachometer-alt')),
                               menuItem("Prediction", tabName = 'Prediction', icon = icon('search')))),
  
  #Tabs layout
  dashboardBody(tags$head(tags$style(HTML('.main-header .logo {font-weight: bold;}'))),
                #Plots tab content
                tabItems(tabItem('Overview',
                                 
                                 # div(tags$img(src="flu-vaccine.jpg", height="500px", width="900px", alt="vaccine"), style="text-align: center;"), imageOutput("vaccineImg"),
                                 div(imageOutput("vaccineImg")),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 strong("Research Question: Can you predict whether people got H1N1 and seasonal flu vaccines using information they shared about their backgrounds, opinions, and health behaviors?"),
                                 br(),
                                 "In this research, we will take a look at vaccination, a key public health measure used to fight infectious diseases.",
                                 "Vaccines provide immunization for individuals, and enough immunization in a community can further reduce the spread of diseases through 'herd immunity.'",
                                 br(),
                                 br(),
                                 strong("Research Motivation:"),
                                 br(),
                                 "As of the launch of this competition, vaccines for the COVID-19 virus are still under development and not yet available.",
                                 "The research will instead revisit the public health response to a different recent major respiratory disease pandemic.",
                                 "Beginning in spring 2009, a pandemic caused by the H1N1 influenza virus, colloquially named 'swine flu,' swept across the world.",
                                 "Researchers estimate that in the first year, it was responsible for between 151,000 to 575,000 deaths globally.",
                                 br(),
                                 br(),
                                 strong("Research Data:"),
                                 br(),
                                 "A vaccine for the H1N1 flu virus became publicly available in October 2009. In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey.",
                                 "Data collected cover a broad scope from whether the respondent receive H1N1 or Seasonal Flu vaccine, social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission",
                                 br(),
                                 br(),
                                 strong("Research Methodology:"),
                                 br(),
                                 # div(tags$img(src="workflow.jpg", height="400px", width="900px", alt="workflow"), style="text-align: center;"), imageOutput("workflowImg"),
                                 div(imageOutput("workflowImg")),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 br(),
                                 strong("Research Goal:"),
                                 br(),
                                 "A better understanding of how these characteristics are associated with personal vaccination patterns can provide guidance for future public health efforts.",
                                 br(),
                                 br(),
                                 strong("Team Member:"),
                                 br(),
                                 "Tan Beng Teong (17110765)",
                                 br(),
                                 "Aw Yeong Fung Mun (17197465)",
                                 br(),
                                 "Chen Yu Qi (17068721)",
                                 br(),
                                 "Chong Li Feng (S2118747)",
                                 br(),
                                 "Kew Jing Sheng (S2021452)",
                                 ),
                         
                         #Dashboard tab content
                         tabItem('Summary',
                                 h1("Overview of Data"),
                                 
                                 # fluidRow(valueBox(26707, "No. of Observations"), valueBoxOutput("noRow"),
                                 #          valueBox(44, "No. of Columns"), valueBoxOutput("noColumn"),
                                 #          valueBox(12435, "No. of Seasonal Flu Vaccinated"), valueBoxOutput("seVaccinated"),
                                 #          valueBox(5674, "No. of H1N1 Flu Vaccinated"), valueBoxOutput("h1n1Vaccinated"),
                                 #          valueBox(8598, "No. of None Vaccinated"), valueBoxOutput("noneVaccinated")),
                                 
                                 fluidRow(valueBoxOutput("noRow"),
                                          valueBoxOutput("noColumn"),
                                          valueBoxOutput("seVaccinated"),
                                          valueBoxOutput("h1n1Vaccinated"),
                                          valueBoxOutput("noneVaccinated")),
                                 
                                 # fluidRow(valueBox(44, "No. of Columns", icon=icon("fa-database")), valueBoxOutput("noColumn")), 
                                 # 
                                 # fluidRow(valueBox(12435, "No. of Seasonal Flu Vaccinated", icon=icon("fa-thumbs-up")), valueBoxOutput("seVaccinated")), 
                                 
                                 # fluidRow(valueBox(12435, "No. of Seasonal Flu Vaccinated", icon=icon("fa-thumbs-up")), valueBoxOutput("seVaccinated"),
                                 #          valueBox(5674, "No. of H1N1 Flu Vaccinated", icon=icon("fa-thumbs-up")), valueBoxOutput("h1n1Vaccinated"),
                                 #          valueBox(8598, "No. of None Vaccinated", icon=icon("fa-exclamation-triangle")), valueBoxOutput("noneVaccinated")),
                                 
                                 # fluidRow(valueBox(8598, "No. of None Vaccinated", icon=icon("fa-exclamation-triangle")), valueBoxOutput("noneVaccinated")),
                                 
                                 tabBox(width = 12, height= "1000px",
                                        tabPanel("Sample of Data",
                                                 selectInput(inputId = "inVaccine", 
                                                             label="Select Between 4 Different Vaccination Status Data",
                                                             choices = impute_train_df$h1n1_seasonal_vaccine, #remove respondent id
                                                             selected = impute_train_df$h1n1_seasonal_vaccine[1]),
                                                 
                                                 fluidRow(align= "center",
                                                          tableOutput("vaccineData"))
                                        ),
                                        tabPanel("Statistics Summary",
                                                 fluidRow(align= "center",
                                                          verbatimTextOutput("vaccineSummary"))
                                        ),
                                        tabPanel("Missing Values Analysis",
                                                 fluidRow(splitLayout( cellWidths = c("50%","50%"), height = "500px",
                                                                       plotOutput("beforeMissing"),
                                                                       plotOutput("afterMissing")))
                                                 )
                                        )
                                 ),
                         
                         tabItem("Visualization",
                                 h1("Exploratory Data Analysis"),
                                 tabBox(
                                        tabPanel("Univariate Variable",
                                                 selectInput(inputId = "univariate", 
                                                             label="Select a Variable",
                                                             choices = colnames(impute_train_df) [-33], #remove respondent id
                                                             selected = colnames(impute_train_df)[1]),
                                                 
                                                 fluidRow(align= "center",
                                                          plotOutput("univariate", width ="65%", height = "450px"))
                                                 ),
                                        tabPanel("Bivariate Variable",
                                                 selectInput(inputId = "bivariate", 
                                                             label="Select a Variable",
                                                             choices = colnames(impute_train_df) [-c(33, 18,34)], #remove respondent id, h1n1_vaccine, seasonal_vaccine
                                                             selected = colnames(impute_train_df)[1]),
                                                 
                                                 fluidRow(splitLayout( cellWidths = c("50%","50%"), height = "500px",
                                                                       plotOutput("bivariate1"),
                                                                       plotOutput("bivariate2")))
                                                 ),
                                        tabPanel("Multivariate Variable",
                                                 selectInput(inputId = "multivariate", 
                                                             label="Select a Variable",
                                                             choices = colnames(impute_train_df) [-c(33, 18,34)],
                                                             selected = colnames(impute_train_df)[1]),
                                                 
                                                 fluidRow(align= "center",
                                                          plotOutput("multivariate", width ="100%",height = "500"))
                                                 ),
                                        tabPanel("Features Correlation",
                                                 sliderInput('sample_sz', 'Sample Size', 
                                                             min=1, max=nrow(impute_train_df), value=min(1000, nrow(impute_train_df)), 
                                                             step=500, round=0),
                                                 selectInput('x_varb', 'X-axis variable', names(impute_train_df)),
                                                 selectInput('y_varb', 'Y-axis variable', names(impute_train_df), names(impute_train_df)[[43]]),
                                                 selectInput('cat_colour', 'Select Categorical variable', names(impute_train_df)),
                                                 selectInput("formula", label="Formula", choices=c("y~x", "y~poly(x,2)", "y~log(x)")),
                                                 
                                                 plotOutput('featureCorrelation', dblclick = "plot_reset")
                                                 )
                                        )
                                 ),
                         
                         #Prediction tab content
                         tabItem('Prediction',
                                 tabBox(width = 12, height= "1500px",
                                        tabPanel("Trained RF Model",
                                                 fluidRow(splitLayout(align= "center",
                                                                      verbatimTextOutput("modelSummary"),
                                                                      plotOutput("rfImportance"))),
                                                 ),
                                        tabPanel("Prediction",
                                                 box(title = 'Categorical variables', status = 'primary', width = 12,
                                                     splitLayout(tags$head(tags$style(HTML(".shiny-split-layout > div {overflow: visible;}"))),
                                                                 cellWidths = c('0%', '13%', '4%', '13%', '4%', '13%', '4%', '13%', '4%', '13%'),
                                                                 selectInput("h1n1_knowledge", "h1n1_knowledge", choices = c(impute_train_df$h1n1_knowledge)),
                                                                 div(),
                                                                 selectInput("behavioral_avoidance", "behavioral_avoidance", choices = c(impute_train_df$behavioral_avoidance)),
                                                                 div(),
                                                                 selectInput("chronic_med_condition", "chronic_med_condition", choices = c(impute_train_df$chronic_med_condition)),
                                                                 div(),
                                                                 selectInput("opinion_h1n1_risk", "opinion_h1n1_risk", choices = c(impute_train_df$opinion_h1n1_risk)),
                                                                 div(),
                                                                 selectInput("opinion_seas_risk", "opinion_seas_risk", choices = c(impute_train_df$opinion_seas_risk)),
                                                                 div(),
                                                     )),
                                                 box(title = 'Categorical variables', status = 'primary', width = 12,
                                                     splitLayout(tags$head(tags$style(HTML(".shiny-split-layout > div {overflow: visible;}"))),
                                                                 cellWidths = c('0%', '13%', '4%', '13%', '4%', '13%', '4%', '13%', '4%', '13%'),
                                                                 selectInput("h1n1_knowledge", "h1n1_knowledge", choices = c(impute_train_df$h1n1_knowledge)),
                                                                 div(),
                                                                 selectInput("behavioral_antiviral_meds", "behavioral_antiviral_meds", choices = c(impute_train_df$behavioral_antiviral_meds)),
                                                                 div(),
                                                                 selectInput("behavioral_face_mask", "behavioral_face_mask", choices = c(impute_train_df$behavioral_face_mask)),
                                                                 div(),
                                                                 selectInput("behavioral_wash_hands", "behavioral_wash_hands", choices = c(impute_train_df$behavioral_wash_hands)),
                                                                 div(),
                                                                 selectInput("behavioral_large_gatherings", "behavioral_large_gatherings", choices = c(impute_train_df$behavioral_large_gatherings)),
                                                                 div(),
                                                     )),
                                                 box(title = 'Categorical variables', status = 'primary', width = 12,
                                                     splitLayout(tags$head(tags$style(HTML(".shiny-split-layout > div {overflow: visible;}"))),
                                                                 cellWidths = c('0%', '13%', '4%', '13%', '4%', '13%', '4%', '13%', '4%', '13%'),
                                                                 selectInput("behavioral_outside_home", "behavioral_outside_home", choices = c(impute_train_df$behavioral_outside_home)),
                                                                 div(),
                                                                 selectInput("behavioral_touch_face", "behavioral_touch_face", choices = c(impute_train_df$behavioral_touch_face)),
                                                                 div(),
                                                                 selectInput("doctor_recc_h1n1", "doctor_recc_h1n1", choices = c(impute_train_df$doctor_recc_h1n1)),
                                                                 div(),
                                                                 selectInput("doctor_recc_seasonal", "doctor_recc_seasonal", choices = c(impute_train_df$doctor_recc_seasonal)),
                                                                 div(),
                                                                 selectInput("child_under_6_months", "child_under_6_months", choices = c(impute_train_df$child_under_6_months)),
                                                                 div(),
                                                     )),
                                                 box(title = 'Categorical variables', status = 'primary', width = 12,
                                                     splitLayout(tags$head(tags$style(HTML(".shiny-split-layout > div {overflow: visible;}"))),
                                                                 cellWidths = c('0%', '13%', '4%', '13%', '4%', '13%', '4%', '13%', '4%', '13%'),
                                                                 selectInput("health_worker", "health_worker", choices = c(impute_train_df$health_worker)),
                                                                 div(),
                                                                 selectInput("health_insurance", "health_insurance", choices = c(impute_train_df$health_insurance)),
                                                                 div(),
                                                                 selectInput("opinion_h1n1_vacc_effective", "opinion_h1n1_vacc_effective", choices = c(impute_train_df$opinion_h1n1_vacc_effective)),
                                                                 div(),
                                                                 selectInput("opinion_h1n1_sick_from_vacc", "opinion_h1n1_sick_from_vacc", choices = c(impute_train_df$opinion_h1n1_sick_from_vacc)),
                                                                 div(),
                                                                 selectInput("opinion_seas_vacc_effective", "opinion_seas_vacc_effective", choices = c(impute_train_df$opinion_seas_vacc_effective)),
                                                                 div(),
                                                     )),
                                                 box(title = 'Categorical variables', status = 'primary', width = 12,
                                                     splitLayout(tags$head(tags$style(HTML(".shiny-split-layout > div {overflow: visible;}"))),
                                                                 cellWidths = c('0%', '13%', '4%', '13%', '4%', '13%', '4%', '13%', '4%', '13%'),
                                                                 selectInput("opinion_seas_sick_from_vacc", "opinion_seas_sick_from_vacc", choices = c(impute_train_df$opinion_seas_sick_from_vacc)),
                                                                 div(),
                                                                 selectInput("household_adults", "household_adults", choices = c(impute_train_df$household_adults)),
                                                                 div(),
                                                                 selectInput("household_children", "household_children", choices = c(impute_train_df$household_children)),
                                                                 div(),
                                                                 selectInput("sex", "sex", choices = c(impute_train_df$sex)),
                                                                 div(),
                                                                 selectInput("marital_status", "marital_status", choices = c(impute_train_df$marital_status)),
                                                                 div(),
                                                     )),
                                                 box(title = 'Categorical variables', status = 'primary', width = 12,
                                                     splitLayout(tags$head(tags$style(HTML(".shiny-split-layout > div {overflow: visible;}"))),
                                                                 cellWidths = c('0%', '13%', '10%', '13%', '10%'),
                                                                 selectInput("rent_or_own", "rent_or_own", choices = c(impute_train_df$rent_or_own)),
                                                                 div(),
                                                                 selectInput("employment_status", "employment_status", choices = c(impute_train_df$employment_status)),
                                                                 div(),
                                                     )),
                                                 #Filters for numeric variables
                                                 box(title = 'Numerical variables', status = 'primary', width = 12,
                                                     splitLayout(cellWidths = c('22%', '4%','21%', '4%', '21%', '4%', '21%'),
                                                                 sliderInput('age_group_labels', 'Age Group', min = 1, max = 5, value = 1),
                                                                 div(),
                                                                 sliderInput('education_labels', 'Education', min = 1, max = 5, value = 1),
                                                                 div(),
                                                                 sliderInput('race_labels', 'Race', min = 1, max = 5, value = 1),
                                                                 div(),
                                                                 sliderInput('income_poverty_labels', 'Income Poverty Group', min = 1, max = 5, value = 1)
                                                     )),
                                                 #Box to display the prediction results
                                                 box(title = 'Prediction result', status = 'success', solidHeader = TRUE, width = 4, height = 260,
                                                     div(h5('Vaccinated Group: 0 = No Vaccination; 1 = Seasonal Flu Vaccinated; 2 = H1N1 Flu Vaccinated; 3 = Vaccinated Both')),
                                                     verbatimTextOutput("value", placeholder = TRUE),
                                                     actionButton('cal','Calculate', icon = icon('calculator'))),
                                                 #Box to display information about the model
                                                 box(title = 'Model explanation', status = 'success', width = 8, height = 260,
                                                     helpText('The following model will predict the respondent have been vaccinated with either H1N1, SEASONAL, NONE or BOTH.'),
                                                     helpText('The name of the dataset used to train the model is "Seasonal & H1N1 Vaccine Data Set", taken from the DRIVENDATA website. The data contains 26,707 observations and 42 attributes.')
                                                     # helpText(sprintf('The prediction is based on a random forest supervised machine learning model. Furthermore, the models deliver a mean absolute error (MAE) of %s Seasonal & H1N1 vaccination, and a root mean squared error (RMSE) of %s Seasonal & H1N1 vaccination.', round(mae_rf, digits = 0), round(rmse_rf, digits = 0)
                                                     #                  )
                                                     #          )
                                                 )
                                        )
                                 )
                                 )
                         )
                )
  )
