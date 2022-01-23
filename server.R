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
impute_train_df <- as.data.frame(impute_train_df)
impute_train_df <- impute_train_df [, order(names(impute_train_df))]

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

cat_input_cols <- c('h1n1_knowledge', 'behavioral_avoidance', 'chronic_med_condition', 'opinion_h1n1_risk', 'opinion_seas_risk')
num_input_cols <- c('age_group_labels', 'education_labels', 'race_labels', 'income_poverty_labels')

# intrain <- createDataPartition(y = impute_train_df$h1n1_seasonal_vaccine, p = 0.70, list = FALSE)
# train <- impute_train_df[intrain,]
# test <- impute_train_df[-intrain,]
set.seed(12345)
# model_rf <- randomForest(y= train[,43], train[,1:42],importance=T, confusion=T, err.rate=T)
model_rf <- readRDS(file = 'model/rf.rda')
y_pred <- predict(model_rf, test, type = "class")
rf.importancePlot <- varImpPlot(model_rf, sort=T, main="Feature Importance Plot for Vaccination (Seasonal & H1N1) Prediction")
# cm = as.matrix(table(Actual = test$h1n1_seasonal_vaccine, Predicted = y_pred))
# cf_rf <- confusionMatrix(y_pred, test$h1n1_seasonal_vaccine)

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

# Define server logic required to draw a histogram
server <- shinyServer(function(input, output) {
  
  dataOut <- reactive({
    impute_train_df
  })
  
  output$noRow <- renderValueBox({
    valueBox(
      26707, "No. of Observations", icon=icon("fa-database"), width=2)
  })
  
  output$noColumn <- renderValueBox({
    valueBox(
      44, "No. of Columns", icon=icon("fa-database"), width=2)
  })
  
  output$seVaccinated <- renderValueBox({
    valueBox(
      12435, "No. of Seasonal Flu Vaccinated", color="yellow", icon=icon("fa-thumbs-up"), width=2)
  })
  
  output$h1n1Vaccinated <- renderValueBox({
    valueBox(
      5674, "No. of H1N1 Flu Vaccinated", color="yellow", icon=icon("fa-thumbs-up"), width=2)
  })
  
  output$noneVaccinated <- renderValueBox({
    valueBox(
      8598, "No. of None Vaccinated", color="red", icon=icon("fa-exclamation-triangle"), width=2)
  })
  
  
  #Prediction model
  #React value when using the action button
  a <- reactiveValues(result = NULL)
  
  observeEvent(input$cal, {
    #Copy of the test data without the dependent variable
    test_pred <- test[-44]
    
    #Dataframe for the single prediction
    values = data.frame(h1n1_concern = input$h1n1_concern, 
                        h1n1_knowledge = input$h1n1_knowledge,
                        behavioral_antiviral_meds = input$behavioral_antiviral_meds,
                        behavioral_avoidance = input$behavioral_avoidance,
                        behavioral_face_mask = input$behavioral_face_mask,
                        behavioral_wash_hands = input$behavioral_wash_hands,
                        behavioral_large_gatherings = input$behavioral_large_gatherings, 
                        behavioral_outside_home = input$behavioral_outside_home, 
                        behavioral_touch_face = input$behavioral_touch_face, 
                        doctor_recc_h1n1 = input$doctor_recc_h1n1,
                        doctor_recc_seasonal = input$doctor_recc_seasonal,
                        child_under_6_months = input$child_under_6_months,
                        health_worker = input$health_worker,
                        health_insurance = input$health_insurance,
                        opinion_h1n1_vacc_effective = input$opinion_h1n1_vacc_effective, 
                        opinion_h1n1_risk = input$opinion_h1n1_risk, 
                        opinion_h1n1_sick_from_vacc = input$opinion_h1n1_sick_from_vacc, 
                        opinion_seas_vacc_effective = input$opinion_seas_vacc_effective,
                        chronic_med_condition = input$chronic_med_condition,
                        opinion_seas_risk = input$opinion_seas_risk,
                        opinion_seas_sick_from_vacc = input$opinion_seas_sick_from_vacc,
                        household_adults = input$household_adults,
                        household_children = input$household_children, 
                        age_group = input$age_group, 
                        education = input$education,
                        race = input$race_labels,
                        sex = input$sex,
                        income_poverty = input$income_poverty_labels,
                        marital_status = input$marital_status,
                        rent_or_own = input$rent_or_own,
                        employment_status = input$employment_status)
    
    #Inclued the values into the new data
    test_pred <- rbind(test_pred,values)
    
    #Single preiction using the randomforest model
    a$result <-  round(predict(model_rf, 
                               newdata = test_pred[nrow(test_pred),]), 
                       digits = 0)
  })
  
  output$value <- renderText({
    #Display the prediction value
    paste(a$result)
  })
  
  output$range <- renderText({
    #Display the range of prediction value using the MAE value
    input$cal
    isolate(sprintf('(%s) - (%s)', 
                    round(a$result - mae_rf, digits = 0), 
                    round(a$result + mae_rf, digits = 0)))
  })
  
  output$vaccineData <- renderTable({
    vaccineFilter <- subset(impute_train_df, impute_train_df$h1n1_seasonal_vaccine==input$inVaccine)
    
    vaccineFilter <- head(vaccineFilter, 20)
  })
  
  output$vaccineSummary <- renderPrint({
    summary(impute_train_df)
  })
  
  output$modelSummary <- renderPrint({
    print(model_rf)
  })
  
  output$rfImportance <- renderPlot({
    print(rf.importancePlot)
  })
  
  output$vaccineImg <- renderImage({
    list(src="assets/flu-vaccine.jpg", height="500px", width="900px", alt="vaccine")
  })
  
  output$workflowImg <- renderImage({
    list(src="assets/workflow.jpg", height="500px", width="900px", alt="workflow")
  })
  
  output$rfImportance <- renderImage({
    list(src="assets/rf_importance.jpg", height="500px", width="900px", alt="rf_importance")
  })
  
  output$featureCorrelation <- renderPlot({
    #Produce scatter plot
    subset_data<-impute_train_df[1:input$sample_sz,]
    
    ggplot(subset_data, aes_string(input$x_varb, input$y_varb))+
      geom_point(aes_string(colour=input$cat_colour))+
      geom_smooth(method="lm",formula=input$formula)
  }, res = 96)
  
  output$beforeMissing <- renderPlot({
    myplot <- ggplot(before_missing.values)
    myplot <- myplot + geom_bar(aes(x = reorder(key, desc(pct)), y = pct, fill=isna), stat = 'identity', alpha=0.8)
    myplot <- myplot + scale_x_discrete(limits = levels)
    myplot <- myplot + scale_fill_manual(name = "", values = c('steelblue', 'tomato3'), labels = c("Present", "Missing"))
    myplot <- myplot + coord_flip()
    myplot <- myplot + labs(title = "Percentage of missing values before Random Forest Imputation", x = 'Variable', y = "% of missing values")
    print(myplot)
  })
  
  output$afterMissing <- renderPlot({
    myplot <- ggplot(after_missing.values)
    myplot <- myplot + geom_bar(aes(x = reorder(key, desc(pct)), y = pct, fill=isna), stat = 'identity', alpha=0.8)
    myplot <- myplot + scale_x_discrete(limits = levels)
    myplot <- myplot + scale_fill_manual(name = "", values = c('steelblue', 'tomato3'), labels = c("Present", "Missing"))
    myplot <- myplot + coord_flip()
    myplot <- myplot + labs(title = "Percentage of missing values after Random Forest Imputation", x = 'Variable', y = "% of missing values")
    print(myplot)
  })
  
  
  # output$Imputation <- renderPlot({impute_train_df = na_mean(train_df)
  # impute_train_df  %>%
  #   summarise_all(list(~is.na(.)))%>%
  #   pivot_longer(everything(),
  #                names_to = "variables", values_to="missing") %>%
  #   count(variables, missing) %>%
  #   ggplot(aes(y=variables,x=n,fill=missing))+
  #   geom_col()})
  
  output$univariate <- renderPlot({
    myplot <- ggplot(impute_train_df)
    myplot <- myplot + aes(x= get(input$univariate), fill = get(input$univariate))+ xlab(input$univariate)
    myplot <- myplot + geom_bar(position= position_dodge(),aes(y= ..count..))
    myplot <- myplot + geom_text(position= position_dodge(width=1),aes(label = scales::percent(round((..count..)/sum(..count..),3))), stat= "count", vjust = -0.5)
    myplot <- myplot + labs(title = paste("Plot of", input$univariate),fill = input$univariate)
    myplot <- myplot + theme_bw()
    print(myplot)
    
  })
  
  output$bivariate1 <- renderPlot({
    myplot <- ggplot(impute_train_df)
    myplot <- myplot + aes(x= get(input$bivariate), fill = as.factor(h1n1_vaccine))+ xlab(input$bivariate)
    myplot <- myplot + geom_bar(position=position_dodge(),aes(y= ..count..))
    myplot <- myplot + geom_text(position= position_dodge(width=1),aes(label = scales::percent(round((..count..)/sum(..count..),3))), stat= "count", vjust = -0.5)
    myplot <- myplot + labs(title = paste("Plot of", input$bivariate, "and H1N1 vaccine"),fill = "H1N1 vaccine")
    myplot <- myplot + theme_bw()
    print(myplot)
  })
  
  output$bivariate2 <- renderPlot({
    myplot <- ggplot(impute_train_df)
    myplot <- myplot + aes(x= get(input$bivariate), fill = as.factor(seasonal_vaccine))+ xlab(input$bivariate)
    myplot <- myplot + geom_bar(position=position_dodge(),aes(y= ..count..))
    myplot <- myplot + geom_text(position= position_dodge(width=1),aes(label = scales::percent(round((..count..)/sum(..count..),3))), stat= "count", vjust = -0.5)
    myplot <- myplot + labs(title = paste("Plot of", input$bivariate, "and seasonal flu vaccine"),fill = "Seasonal flu vaccine")
    myplot <- myplot + theme_bw()
    print(myplot)
    
  })
  
  
  output$multivariate <- renderPlot({
    
    Flu_table <- impute_train_df %>% data.frame() %>% group_by(!!sym(input$multivariate),h1n1_vaccine, seasonal_vaccine) %>%
      count()
    print(Flu_table)
    
    myplot <- ggplot(as.data.frame(Flu_table), aes(axis1 = !!sym(input$multivariate), axis2 = h1n1_vaccine,y = n))+
      geom_alluvium(aes(fill = seasonal_vaccine))+
      geom_stratum()+
      geom_text(stat = "stratum",aes(label = after_stat(stratum))) +
      scale_x_discrete(limits = c(input$multivariate, "h1n1 vaccine"),expand = c(.1, .1)) +
      labs(title = "H1N1 Vaccination",
           subtitle = paste("stratified by seasonal flu vaccine and",input$multivariate),
           fill = "seasonal flu vaccine",
           y = "Frequency") +
      theme_minimal()
    
    
    
    print(myplot)
    
  })
  
})

server <- function(input, output){
  
}