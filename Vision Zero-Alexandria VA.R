 # Errol Schwartz (G01247477)
# Shima Mohebbi
# STAT 468: Term project
# 12 May 2022

#______________________________________________________________________________

# DATA SOURCE -----------------------------------------------------------------

# Data source: Virginia Department of Transportation (VDOT)
# Data set link: https://www.virginiaroads.org/
# Data set descriptions: https://www.virginiadot.org/business/VDOT_Crash_Data_Manual_Nov2017.pdf

# LIBRARIES --------------------------------------------------------------------
library(car) # multicollinearity check: variance inflation factor, vif()
library(caret) # variable importance with decision trees, train/test splits with createDataPartition()
library(deepnet) # Stacked auto-encoder deep neural network: method = 'dnn'
library(dplyr) # filtering: filter(), select_if()
library(e1071) # support vector machine (svm)
library(earth) # for MARS
library(EnvStats) # transform data with boxcox()
library(forecast) # time series forecasting with neural net: nnar()
library(ggplot2) # visualisation
library(lubridate) # aggregating target variable values over custom date formats: floor_date()
library(M3) # combine date and time columns into datetime object: combine.date.and.time()
library(Metrics) # model performance evaluation
library(modelr) # add predictions to source data frame: add_predictions()
library(modelsummary) # comparing models
library(moments) # checking data distribution: skewness(), kurtosis()
library(naniar) # tabulate and visualise missing values: miss_var_summary(), gg_miss_var()
library(neuralnet) # for ANN
library(NeuralNetTools) # plotting neural nets
library(party) # tree visualisation
library(partykit) # tree visualisation
library(rattle) # fancy tree plot
library(rpart) # decision tree modeling
library(rpart.plot) # enhanced tree plots
library(stringr) # individual columns for each value in single cell: str_split_fixed()
library(xts) # convertion to time-series objects: xts()
# ------------------------------------------------------------------------------
# IMPORT and SUBSET DATA -------------------------------------------------------
# Import: Virginia Crashes data set ____________________________________________
file.choose()
df <- read.csv("/Users/schwartz/Library/Mobile Documents/com~apple~CloudDocs/GMU/4. Spring 2022/SYST 468/Term project/Virginia_Crashes.csv", header = T)
glimpse(df) # sanity check
dim(df) # (882957,40)

# Subset: Extract City of Alexandria ___________________________________________
# City of Alexandria jurisdiction code -- PHYSICAL_JURIS == "100. Alexandria"
alex.100 <- df[df$PHYSICAL_JURIS == "100.Alexandria",]
glimpse(alex.100) # sanity check. Dimensions = (10429,40)

# A visual inspection shows that CRASH_SEVERITY and Driverinjurytype have more 
# values than Passinjurytype and Pedinjurytype.

# Data set description (on website) lists 'A. Severe Injury'. So, filter alex.100
# by crash severity (which will inevitably include Driverinjurytype):
filter(alex.100, CRASH_SEVERITY == "A.Severe Injury")

# To avoid wasting time scrolling through data set, use grep() function 
#to search for deaths or fatalities (and variances of death and fatality):
alex.100[grep("Death|Fatal|Fatality", alex.100$CRASH_SEVERITY, ignore.case = T),]
# "K.Fatal Injury" is how a traffic death is classified. 

# New data frame with only severe injury and fatality:
alex <- filter(alex.100, CRASH_SEVERITY == "A.Severe Injury"|CRASH_SEVERITY == "K.Fatal Injury")
glimpse(alex) # sanity check. Dimensions = (282,40)

# Confirm that severe and fatal injuries are in alex:
alex[alex$CRASH_SEVERITY == 'A.Severe Injury',] # yes
alex[alex$CRASH_SEVERITY == 'K.Fatal Injury',] # yes

# Successfully filtered original data frame, df, down to a new data frame, alex, 
# with target observations, severe injury and fatality.
head(alex) # sanity check

# DATA UNDERSTANDING -----------------------------------------------------------

# Missing values _______________________________________________________________
head(as.data.frame(miss_var_summary(alex))) 
# Roadway Network System Milepost (RNS_MP) has the only missing value
# miss_var_summary() returns tibble. Convert to data frame and call head()
gg_miss_var(alex) # visualise missing values

# DATA PREPARATION -------------------------------------------------------------

# Treat missing values _________________________________________________________
# Where in RNS_MP is this missing value? Which row has this missing value?
which(is.na(alex$RNS_MP), arr.ind = T) # row 83
str(alex$RNS_MP) # numeric data type. 

# Imputation
alex$RNS_MP[is.na(alex$RNS_MP)] <- mean(alex$RNS_MP, na.rm = T)
alex$RNS_MP[83] # new value = 32.21445

alex$RNS_MP
sum(is.na(alex$RNS_MP)) # missing values sanity check. No more missing values

##----------------- DATA CLEANING and FEATURE ENGINEERING --------------------##
# General housekeeping:
#   1. treat features with more than one data type in a cell
#   2. convert features to relevant data types
#   3. perform feature engineering

glimpse(alex) # data types overview

# Note: each variable has different cleaning requirements. I organised each
# variable into its own section where I'll apply targeted processing.

## _________________________ CRASH_MILITARY_TIME _______________________________
str(alex$CRASH_MILITARY_TM) # sanity check

## To-do:
##    1. change data type from numeric to time series

# Format time string to four digits
alex$CRASH_MILITARY_TM <- sprintf('%04d', alex$CRASH_MILITARY_TM) 
# Format time string in H:M:S format without time zone information
alex$CRASH_MILITARY_TM <- format(strptime(alex$CRASH_MILITARY_TM, format = '%H%M'), format = '%H:%M:%S', usetz = F) 
str(alex$CRASH_MILITARY_TM)

## _____________________________ Passage _______________________________________
str(alex$Passage) # sanity check
unique(alex$Passage) # unique number 

## To-do:
##    1. remove numbers, only keep words / descriptors
##    2. deal with mixed data types (in each cell)
##    3. split multiple values in a single cell across separate columns
##    4. drop original column

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Passage), sep = ';')) # 6

alex$Passage <- gsub("[;]"," ", alex$Passage) # remove semicolon; replace with empty space
head(alex$Passage) # sanity check
alex[c('Passenger1_age','Passenger2_age',
       'Passenger3_age','Passenger4_age',
       'Passenger5_age','Passenger6_age')] <- str_split_fixed(alex$Passage,' ', 6) # split space-delimited values into new columns

# Sanity checks
alex$Passenger1_age
alex$Passenger2_age
alex$Passenger3_age
alex$Passenger4_age
alex$Passenger5_age
alex$Passenger6_age
# Sanity checks show empty strings

# Change new columns to numeric/integer data type
alex[c('Passenger1_age','Passenger2_age',
       'Passenger3_age','Passenger4_age',
       'Passenger5_age','Passenger6_age')]<- sapply(alex[c('Passenger1_age','Passenger2_age',
                                                           'Passenger3_age','Passenger4_age',
                                                           'Passenger5_age','Passenger6_age')], as.numeric)
# Sanity check
head(subset(alex, select = c('Passenger1_age','Passenger2_age',
                             'Passenger3_age','Passenger4_age',
                             'Passenger5_age','Passenger6_age')))

# Data type sanity check
str(subset(alex, select = c('Passenger1_age','Passenger2_age',
                            'Passenger3_age','Passenger4_age',
                            'Passenger5_age','Passenger6_age')))

# Replace NA values with 0
alex[c('Passenger1_age','Passenger2_age',
       'Passenger3_age','Passenger4_age',
       'Passenger5_age','Passenger6_age')][is.na(alex[c('Passenger1_age','Passenger2_age',
                                                        'Passenger3_age','Passenger4_age',
                                                        'Passenger5_age','Passenger6_age')])] <- 0

# Data type sanity check (numeric)
str(subset(alex, select = c('Passenger1_age','Passenger2_age',
                            'Passenger3_age','Passenger4_age',
                            'Passenger5_age','Passenger6_age')))

## ______________________________ Pedage _______________________________________
str(alex$Pedage) # sanity check

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column 

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Pedage), sep = ';')) # 2
alex$Pedage <- gsub("[;]"," ", alex$Pedage) # remove semicolon; replace with empty space
head(alex$Pedage) # sanity check
alex[c('Pedestrian1_age', 'Pedestrian2_age')] <- str_split_fixed(alex$Pedage,' ', 2) # split space-delimited values into new columns
alex$Pedestrian1_age # sanity check. Good, but there are NA values. Not a problem…
alex$Pedestrian2_age # sanity check. NA values as well. Sit tight…
alex[c('Pedestrian1_age', 'Pedestrian2_age')] <- sapply(alex[c('Pedestrian1_age', 'Pedestrian2_age')], as.numeric) # change data types simultaneously
str(alex[c('Pedestrian1_age', 'Pedestrian2_age')]) # data types sanity check (numeric)
alex[c('Pedestrian1_age', 'Pedestrian2_age')][is.na(alex[c('Pedestrian1_age', 'Pedestrian2_age')])] <- 0 # Fill NA values
str(alex$Pedestrian1_age)
sum(is.na(alex$Ped1_age)) # no NA values
sum(is.na(alex$Ped2_age)) # no NA values


## ____________________________ CRASH_DT _______________________________________
str(alex$CRASH_DT) # sanity check

## To-do:
##    1. change data type from chr datetime

alex$CRASH_DT <- as.Date(alex$CRASH_DT, '%Y/%m/%d')
str(alex$CRASH_DT) # data type sanity check (Date)

## __________________________ COLLISION_TYPE ___________________________________
str(alex$COLLISION_TYPE) # chr
unique(alex$COLLISION_TYPE) # unique crash descriptors

## To-do:
##    1. remove numbers and punctuation; keep descriptors

# ('.*? ') meaning: look for any character zero or more times (.*) up until
# the first white space. 
#(?) meaning: make the search lazy and stop after the first white space
alex$COLLISION_TYPE <- sub('.*?.. ','',alex$COLLISION_TYPE) # drop everything that's not a word
alex$COLLISION_TYPE[alex$COLLISION_TYPE == 'Ped'] <- 'Pedestrian' # rename Ped as 'Pedestrian'
unique(alex$COLLISION_TYPE) # sanity check
str(alex$COLLISION_TYPE) # data type sanity check (character)

## __________________________ CRASH_SEVERITY ___________________________________
str(alex$CRASH_SEVERITY) # sanity check
unique(alex$CRASH_SEVERITY) # unique descriptors

## To-do:
##    1. remove letters and punctuation; only keep words / descriptors

alex$CRASH_SEVERITY <- sub('.*?..','',alex$CRASH_SEVERITY) # only keep words; drop everything else
unique(alex$CRASH_SEVERITY) # sanity check
str(alex$CRASH_SEVERITY) # data type sanity check (character)

## __________________________ VDOT_DISTRICT ____________________________________
str(alex$VDOT_DISTRICT) # sanity check

## To-do:
##    1. remove letters and punctuation; only keep words / descriptors
 
alex$VDOT_DISTRICT <- sub('.*?..','',alex$VDOT_DISTRICT) # only keep words; drop everything else
str(alex$VDOT_DISTRICT) # data type sanity check (character)

## ______________________________ RTE_NM _______________________________________
str(alex$RTE_NM) # sanity check

## To-do:
##    1. remove white space in route names

alex$RTE_NM <- str_squish(alex$RTE_NM) # remove leading, middle, and trailing white space
unique(alex$RTE_NM) # sanity check
str(alex$RTE_NM) # data type sanity check (character)

## ____________________________ Ownership ______________________________________
str(alex$Ownership) # sanity check

## To-do:
##    1. remove numbers; only keep words / descriptors

alex$Ownership <- sub('.*?..','',alex$Ownership) # only keep words; drop everything else
unique(alex$Ownership) # sanity check
str(alex$Ownership) # data type sanity check (character)

## _________________________ Driver_Action_Type_Cd _____________________________
str(alex$Driver_Action_Type_Cd) # sanity check

## To-do:
##    1. remove numbers, only keep words / descriptors
##    2. deal with mixed data types (in each cell)
##    3. split multiple values in a single cell across separate columns
##    4. drop original column

alex$Driver_Action_Type_Cd <- gsub('[0-9.]+','',alex$Driver_Action_Type_Cd) # remove all numbers and trailing periods
head(alex$Driver_Action_Type_Cd) # sanity check
alex$Driver_Action_Type_Cd <- str_trim(alex$Driver_Action_Type_Cd) # trim leading white space
head(alex$Driver_Action_Type_Cd) # sanity check

# Feature engineering:
# Determine how many new columns to create. Count values in each cell,
# then find max value. Max value will determine number of new columns
max(count.fields(textConnection(alex$Driver_Action_Type_Cd), sep = ';')) # 6
alex[c('Driver1_action','Driver2_action',
       'Driver3_action','Driver4_action',
       'Driver5_action',
       'Driver6_action')] <- str_split_fixed(alex$Driver_Action_Type_Cd,';', 6) # split semicolon-delimited values into 6 new columns

# Sanity checks
alex$Driver1_action
alex$Driver2_action
alex$Driver3_action
alex$Driver4_action
alex$Driver5_action
alex$Driver6_action

# Sanity checks reveal leading white spaces, n/a values, empty strings

# Trim white space across all six driver action columns
alex[c('Driver1_action','Driver2_action',
       'Driver3_action','Driver4_action',
       'Driver5_action','Driver6_action')] <- lapply(alex[c('Driver1_action','Driver2_action',
                                                            'Driver3_action','Driver4_action',
                                                            'Driver5_action','Driver6_action')], trimws)

# Replace n/a and empty strings with 'Not Provided'
## Note: the code below makes more sense as a single line. I wrote it this way
##      to prevent scrolling too far to the right
alex[c('Driver1_action','Driver2_action',
       'Driver3_action','Driver4_action',
       'Driver5_action','Driver6_action')][(alex[c('Driver1_action','Driver2_action',
                                                  'Driver3_action','Driver4_action',
                                                  'Driver5_action','Driver6_action')] == 'n/a')|
                                           (alex[c('Driver1_action','Driver2_action',
                                                  'Driver3_action','Driver4_action',
                                                  'Driver5_action','Driver6_action')] == '')] <- 'Not Provided' 

# Sanity check
head(subset(alex, select = c('Driver1_action','Driver2_action',
                        'Driver3_action','Driver4_action',
                        'Driver5_action','Driver6_action')))

# Data type sanity check (character)
str(subset(alex, select = c('Driver1_action','Driver2_action',
                            'Driver3_action','Driver4_action',
                            'Driver5_action','Driver6_action')))

## ________________________ VEHICLE_BODY_TYPE_CD _______________________________
unique(alex$VEHICLE_BODY_TYPE_CD) # sanity check

## To-do:
##    1. remove numbers, only keep words / descriptors
##    2. deal with mixed data types (in each cell)
##    3. split multiple values in a single cell across separate columns
##    4. drop original column

alex$VEHICLE_BODY_TYPE_CD <- gsub('[0-9.]+','',alex$VEHICLE_BODY_TYPE_CD) # remove all numbers and trailing periods
head(alex$VEHICLE_BODY_TYPE_CD) # sanity check
alex$VEHICLE_BODY_TYPE_CD <- str_trim(alex$VEHICLE_BODY_TYPE_CD) # trim leading white space
head(alex$VEHICLE_BODY_TYPE_CD) # sanity check
unique(alex$VEHICLE_BODY_TYPE_CD) # sanity check

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$VEHICLE_BODY_TYPE_CD), sep = ';')) # 6
alex[c('Vehicle1_type','Vehicle2_type',
       'Vehicle3_type','Vehicle4_type',
       'Vehicle5_type',
       'Vehicle6_type')] <- str_split_fixed(alex$VEHICLE_BODY_TYPE_CD,';', 6) # split semicolon-delimited values into 6 new columns

# Sanity checks
alex$Vehicle2_type
alex$Vehicle2_type
alex$Vehicle3_type
alex$Vehicle4_type
alex$Vehicle5_type
alex$Vehicle6_type

# Sanity checks show leading white spaces, n/a values, empty strings

# Trim white space across all six vehicle types columns
alex[c('Vehicle1_type','Vehicle2_type',
       'Vehicle3_type','Vehicle4_type',
       'Vehicle5_type','Vehicle6_type')] <- lapply(alex[c('Vehicle1_type','Vehicle2_type',
                                                          'Vehicle3_type','Vehicle4_type',
                                                          'Vehicle5_type','Vehicle6_type')], trimws)

# Replace n/a and empty strings with 'Not Provided'
alex[c('Vehicle1_type','Vehicle2_type',
       'Vehicle3_type','Vehicle4_type',
       'Vehicle5_type','Vehicle6_type')][(alex[c('Vehicle1_type','Vehicle2_type',
                                                 'Vehicle3_type','Vehicle4_type',
                                                 'Vehicle5_type','Vehicle6_type')] == 'n/a')|
                                             (alex[c('Vehicle1_type','Vehicle2_type',
                                                     'Vehicle3_type','Vehicle4_type',
                                                     'Vehicle5_type','Vehicle6_type')] == '')] <- 'Not Provided' 

# Sanity check
head(subset(alex, select = c('Vehicle1_type','Vehicle2_type',
                             'Vehicle3_type','Vehicle4_type',
                             'Vehicle5_type','Vehicle6_type')))

# Data type sanity check (character)
str(subset(alex, select = c('Vehicle1_type','Vehicle2_type',
                             'Vehicle3_type','Vehicle4_type',
                             'Vehicle5_type','Vehicle6_type')))

## _________________________ LIGHT_CONDITION ___________________________________ 
unique(alex$LIGHT_CONDITION) # sanity check

## To-do:
##    1. remove numbers; only keep words / descriptors

alex$LIGHT_CONDITION <- sub('.*? ','',alex$LIGHT_CONDITION) # drop everything that's not a word
unique(alex$LIGHT_CONDITION) # sanity check
str(alex$LIGHT_CONDITION) # data type sanity check (character)

## _______________________ ROADWAY_SURFACE_COND ________________________________
unique(alex$ROADWAY_SURFACE_COND) # sanity check

## To-do:
##    1. remove numbers; only keep words / descriptors

alex$ROADWAY_SURFACE_COND <- sub('.*? ','',alex$ROADWAY_SURFACE_COND) # drop everything that's not a word
unique(alex$ROADWAY_SURFACE_COND) # sanity check
str(alex$ROADWAY_SURFACE_COND) # data type sanity check (character)

## ________________________ WEATHER_CONDITION __________________________________
unique(alex$WEATHER_CONDITION) # sanity check

## To-do:
##    1. remove numbers; only keep words / descriptors

alex$WEATHER_CONDITION <- sub('.*? ','',alex$WEATHER_CONDITION) # drop everything that's not a word
unique(alex$WEATHER_CONDITION) # sanity check
str(alex$WEATHER_CONDITION) # data type sanity check (character)

## _________________________ Work_Zone_Related _________________________________
unique(alex$Work_Zone_Related) # sanity check

## To-do:
##    1. remove numbers; only keep words / descriptors

alex$Work_Zone_Related <- sub('.*? ','',alex$Work_Zone_Related) # drop everything that's not a word
unique(alex$Work_Zone_Related) # sanity check
str(alex$Work_Zone_Related) # data type sanity check (character)

## ___________________________ Driverage _______________________________________
unique(alex$Driverage) # sanity check
str(alex$Driverage) # data type sanity check

## To-do:
##    1. remove numbers, only keep words / descriptors
##    2. deal with mixed data types (in each cell)
##    3. split multiple values in a single cell across separate columns
##    4. drop original column

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Driverage), sep = ';')) # 6

alex$Driverage <- gsub("[;]"," ", alex$Driverage) # remove semicolon; replace with empty space
head(alex$Driverage) # sanity check
alex[c('Driver1_age','Driver2_age',
       'Driver3_age','Driver4_age',
       'Driver5_age','Driver6_age')] <- str_split_fixed(alex$Driverage,' ', 6) # split space-delimited values into new columns

# Sanity checks
alex$Driver1_age
alex$Driver2_age
alex$Driver3_age
alex$Driver4_age
alex$Driver5_age
alex$Driver6_age
# Sanity checks show empty strings

# Change new columns to numeric/integer data type
alex[c('Driver1_age','Driver2_age',
       'Driver3_age','Driver4_age',
       'Driver5_age','Driver6_age')]<- sapply(alex[c('Driver1_age','Driver2_age',
                                                 'Driver3_age','Driver4_age',
                                                 'Driver5_age','Driver6_age')], as.numeric)
# Sanity check
head(subset(alex, select = c('Driver1_age','Driver2_age',
                             'Driver3_age','Driver4_age',
                             'Driver5_age','Driver6_age')))

# Data type sanity check
str(subset(alex, select = c('Driver1_age','Driver2_age',
                             'Driver3_age','Driver4_age',
                             'Driver5_age','Driver6_age')))

# Replace NA values with 0
alex[c('Driver1_age','Driver2_age',
       'Driver3_age','Driver4_age',
       'Driver5_age','Driver6_age')][is.na(alex[c('Driver1_age','Driver2_age',
                                              'Driver3_age','Driver4_age',
                                              'Driver5_age','Driver6_age')])] <- 0
alex <- subset(alex, select = -c(Driverage))

# Sanity check
head(subset(alex, select = c('Driver1_age','Driver2_age',
                             'Driver3_age','Driver4_age',
                             'Driver5_age','Driver6_age')))

# Data type sanity check (numeric)
str(subset(alex, select = c('Driver1_age','Driver2_age',
                             'Driver3_age','Driver4_age',
                             'Driver5_age','Driver6_age')))

## ___________________________ Driverinjurytype ________________________________
unique(alex$Driverinjurytype) # sanity check

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column

alex$Driverinjurytype <- gsub("[;]"," ", alex$Driverinjurytype) # remove semicolon; replace with empty space

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Driverinjurytype), sep = ' ')) # 6
alex[c('Driver1_injury','Driver2_injury',
       'Driver3_injury','Driver4_injury',
       'Driver5_injury','Driver6_injury')] <- str_split_fixed(alex$Driverinjurytype,' ', 6) # split space-delimited values into new columns

# Sanity checks
alex$Driver1_injury
alex$Driver2_injury
alex$Driver3_injury
alex$Driver4_injury
alex$Driver5_injury
alex$Driver6_injury
# Sanity checks show empty strings, "NA" values

# Replace NA values with 0
alex[c('Driver1_injury','Driver2_injury',
       'Driver3_injury','Driver4_injury',
       'Driver5_injury','Driver6_injury')][(alex[c('Driver1_injury','Driver2_injury',
                                                   'Driver3_injury','Driver4_injury',
                                                   'Driver5_injury','Driver6_injury')] == 'NA')|
                                             (alex[c('Driver1_injury','Driver2_injury',
                                                    'Driver3_injury','Driver4_injury',
                                                    'Driver5_injury','Driver6_injury')] == '')] <- 'Not Provided' 
# Sanity check
head(subset(alex, select = c('Driver1_injury','Driver2_injury',
                             'Driver3_injury','Driver4_injury',
                             'Driver5_injury','Driver6_injury')))

# Data type sanity check (character)
str(subset(alex, select = c('Driver1_injury','Driver2_injury',
                            'Driver3_injury','Driver4_injury',
                            'Driver5_injury','Driver6_injury')))

## _____________________________ Drivergen _____________________________________
unique(alex$Drivergen) # sanity check

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column

alex$Drivergen <- gsub("[;]"," ", alex$Drivergen) # remove semicolon; replace with empty space

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(unique(alex$Drivergen)))) # 6
alex$Drivergen <- str_trim(alex$Drivergen) # trim leading white space
alex[c('Driver1_gender','Driver2_gender',
       'Driver3_gender','Driver4_gender',
       'Driver5_gender',
       'Driver6_gender')] <- str_split_fixed(alex$Drivergen,' ', 6) # split space-delimited values into 6 new columns

# Sanity checks
alex$Driver1_gender
alex$Driver2_gender
alex$Driver3_gender
alex$Driver4_gender
alex$Driver5_gender
alex$Driver6_gender

# Sanity checks reveal n/a values, empty strings

# Replace n/a and empty strings with 'Not Provided'
alex[c('Driver1_gender','Driver2_gender',
       'Driver3_gender','Driver4_gender',
       'Driver5_gender','Driver6_gender')][(alex[c('Driver1_gender','Driver2_gender',
                                                   'Driver3_gender','Driver4_gender',
                                                   'Driver5_gender','Driver6_gender')] == 'n/a')|
                                             (alex[c('Driver1_gender','Driver2_gender',
                                                     'Driver3_gender','Driver4_gender',
                                                     'Driver5_gender','Driver6_gender')] == 'n/a  n/a')|
                                             (alex[c('Driver1_gender','Driver2_gender',
                                                     'Driver3_gender','Driver4_gender',
                                                     'Driver5_gender','Driver6_gender')] == ' n/a  n/a  n/a')|
                                             (alex[c('Driver1_gender','Driver2_gender',
                                                     'Driver3_gender','Driver4_gender',
                                                     'Driver5_gender','Driver6_gender')] == '')] <- 'Not Provided' 

# Sanity check
head(subset(alex, select = c('Driver1_gender','Driver2_gender',
                             'Driver3_gender','Driver4_gender',
                             'Driver5_gender','Driver6_gender')))

# Sanity check
head(subset(alex, select = c('Driver1_gender','Driver2_gender',
                             'Driver3_gender','Driver4_gender',
                             'Driver5_gender','Driver6_gender')))

# Data type sanity check (character)
str(subset(alex, select = c('Driver1_gender','Driver2_gender',
                            'Driver3_gender','Driver4_gender',
                            'Driver5_gender','Driver6_gender')))

## _____________________________ Passgen _______________________________________
unique(alex$Passgen) # sanity check

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(unique(alex$Passgen)), sep = ';')) # 6
alex$Passgen <- str_trim(alex$Passgen) # trim leading white space
alex[c('Passenger1_gender','Passenger2_gender',
       'Passenger3_gender','Passenger4_gender',
       'Passenger5_gender',
       'Passenger6_gender')] <- str_split_fixed(alex$Passgen,';', 6) # split space-delimited values into 6 new columns

# Sanity checks
alex$Passenger1_gender
alex$Passenger2_gender
alex$Passenger3_gender
alex$Passenger4_gender
alex$Passenger5_gender
alex$Passenger6_gender

# Sanity checks reveal empty strings

# Replace empty strings with 'Not Provided'
alex[c('Passenger1_gender','Passenger2_gender',
       'Passenger3_gender','Passenger4_gender',
       'Passenger5_gender',
       'Passenger6_gender')][(alex[c('Passenger1_gender','Passenger2_gender',
                                     'Passenger3_gender','Passenger4_gender',
                                     'Passenger5_gender',
                                     'Passenger6_gender')] == '')] <- 'Not Provided' 

# Sanity check
head(subset(alex, select = c('Passenger1_gender','Passenger2_gender',
                             'Passenger3_gender','Passenger4_gender',
                             'Passenger5_gender','Passenger6_gender')))

str(alex)

## ___________________________ Passinjurytype __________________________________
unique(alex$Passinjurytype)

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column

alex$Passinjurytype <- gsub("[;]"," ", alex$Passinjurytype) # remove semicolon; replace with empty space

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Passinjurytype), sep = ' ')) # 6
alex[c('Passenger1_injury','Passenger2_injury',
       'Passenger3_injury','Passenger4_injury',
       'Passenger5_injury','Passenger6_injury')] <- str_split_fixed(alex$Passinjurytype,' ', 6) # split space-delimited values into new columns

# Sanity checks
alex$Passenger1_injury
alex$Passenger2_injury
alex$Passenger3_injury
alex$Passenger4_injury
alex$Passenger5_injury
alex$Passenger6_injury
# Sanity checks show empty strings

# Replace empty strings with 'Not Provided'
alex[c('Passenger1_injury','Passenger2_injury',
       'Passenger3_injury','Passenger4_injury',
       'Passenger5_injury',
       'Passenger6_injury')][(alex[c('Passenger1_injury','Passenger2_injury',
                                     'Passenger3_injury','Passenger4_injury',
                                     'Passenger5_injury','Passenger6_injury')] == '')] <- 'Not Provided' 
                                             
# Sanity check
head(subset(alex, select = c('Passenger1_injury','Passenger2_injury',
                             'Passenger3_injury','Passenger4_injury',
                             'Passenger5_injury','Passenger6_injury')))

# Data type sanity check (character)
str(subset(alex, select = c('Passenger1_injury','Passenger2_injury',
                            'Passenger3_injury','Passenger4_injury',
                            'Passenger5_injury','Passenger6_injury')))

## ___________________________ Pedinjurytype ___________________________________
unique(alex$Pedinjurytype)

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column

alex$Pedinjurytype <- gsub("[;]"," ", alex$Pedinjurytype) # remove semicolon; replace with empty space

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Pedinjurytype), sep = ' ')) # 2
alex[c('Pedestrian1_injury','Pedestrian2_injury')] <- str_split_fixed(alex$Pedinjurytype,' ', 2) # split space-delimited values into new columns

# Sanity checks
alex$Pedestrian1_injury
alex$Pedestrian2_injury

# Sanity checks show empty strings

# Replace empty strings with 'Not Provided'
alex[c('Pedestrian1_injury',
       'Pedestrian2_injury')][alex[c('Pedestrian1_injury',
                                     'Pedestrian2_injury')] == ''] <- 'Not Provided' 

# Sanity check
head(subset(alex, select = c('Pedestrian1_injury',
                             'Pedestrian2_injury')))

# Data type sanity check (character)
str(subset(alex, select = c('Pedestrian1_injury',
                            'Pedestrian2_injury')))

## ______________________________ Pedgen _______________________________________
unique(alex$Pedgen)

## To-do:
##    1. deal with mixed data types (in each cell)
##    2. split multiple values in a single cell across separate columns
##    3. drop original column

alex$Pedgen <- gsub("[;]"," ", alex$Pedgen) # remove semicolon; replace with empty space

# Feature engineering: determine how many columns to create
max(count.fields(textConnection(alex$Pedgen), sep = ' ')) # 2
alex[c('Pedestrian1_gender','Pedestrian2_gender')] <- str_split_fixed(alex$Pedgen,' ', 2) # split space-delimited values into new columns

# Sanity checks
alex$Pedestrian1_gender
alex$Pedestrian2_gender

# Sanity checks show n/a values and empty strings

# Replace empty strings with 'Not Provided'
alex[c('Pedestrian1_gender',
       'Pedestrian2_gender')][(alex[c('Pedestrian1_gender',
                                     'Pedestrian2_gender')] == '')|
                              (alex[c('Pedestrian1_gender',
                                     'Pedestrian2_gender')] == 'n/a')] <- 'Not Provided' 

# Sanity check
head(subset(alex, select = c('Pedestrian1_gender',
                             'Pedestrian2_gender')))

# Data type sanity check (character)
str(subset(alex, select = c('Pedestrian1_gender',
                            'Pedestrian2_gender')))

# _______________ FEATURE ENGINEERING (FE): target features ____________________
# Objective: create target variables 'Serious_Injuries' and 'Fatalities'

## To-do:
##    1. filter alex data frame by Driver injury type, Passenger injury type, 
##       and Pedestrian injury type
##    2. for 'Serious_Injuries' feature, filter for 'A'
##    3. for 'Fatalities' feature, filter for 'K'
## 'Serious_Injuries" and 'Fatalities' will be the sum totals of severe and
## fatal injuries for drivers, passengers, and pedestrians for each time step

# Select Driver, Passenger, and Pedestrian injury features
key.features <- c("Driver1_injury",'Driver2_injury','Driver3_injury',
                     'Driver4_injury','Driver5_injury','Driver6_injury',
                     'Passenger1_injury','Passenger2_injury','Passenger3_injury',
                     'Passenger4_injury','Passenger5_injury','Passenger6_injury',
                     'Pedestrian1_injury','Pedestrian2_injury')

# Create 'Serious_Injuries' feature
alex$Serious_Injuries <- rowSums(alex[key.features] == 'A')
head(alex$Serious_Injuries, 15) # sanity check

# Create 'Fatalities' feature
alex$Fatalities <- rowSums(alex[key.features] == 'K')
head(alex$Fatalities, 15) # sanity check

head(subset(alex, select = c(Serious_Injuries, Fatalities)),20) # sanity check
dim(alex) # (282,97)
str(alex)

# Create two data frames from alex, each one with only one target feature
# So alex.serious will have 'Severe_Injury' as its target feature without the 
# 'Fatalities' feature; alex.fatal will have 'Fatalities' as its target feature
# without the 'Severe_Injuries' feature.

## ___________________ FE: Number of vehicles / Number of drivers ______________
str(alex$Num_Vehicles) # sanity check

## To-do:
##    1. change data type from character to integer
##    2. count number of values in each cell
##    3. drop original column

head(alex$VEHICLENUMBER) # separator is a semicolon.
alex$Total_Drivers <- count.fields(textConnection(alex$VEHICLENUMBER), sep = ';') # count number of values in each cell
alex$Total_Drivers # sanity check
head(alex)
dim(alex) # (282,98)

glimpse(alex)

# ______________________ FE: Dropping excess features __________________________
## The following features will be dropped since the data set (alex) is already
## either filtered by one or more of these features (jurisdiction, district, etc.), 
## the feature is an administrative indicator (document number, case ID, etc.),
## hyperlinks (for diagrams), etc.

str(alex)
alex <- subset(alex, select = c(CRASH_DT,COLLISION_TYPE,RTE_NM,
                        LIGHT_CONDITION,ROADWAY_SURFACE_COND,
                        Work_Zone_Related, Driver1_action, Vehicle1_type,
                        Driver1_age,Driver1_gender,Total_Drivers,Serious_Injuries,Fatalities))

colnames(alex) <- c('CrashYear','CollisionType','RouteName','LightCondition',
                    'RoadSurfaceCondition','WorkZone','DriverAction',
                    'VehicleType','DriverAge','DriverGender','TotalDrivers',
                    'SeriousInjuries','Fatalities')
glimpse(alex) # sanity check
alex <- alex[order(alex$CrashYear),]

# _________________________ OUTLIERS and DISTRIBUTION __________________________
## To-do:
##    1. split alex data set into numeric and character data frames
##    2. get summary statistics, outlier analysis, etc. on numeric data frame

str(alex)
# Split data
alex.num1 <- select_if(alex, is.numeric) # select only numeric features

# Sanity checks
glimpse(alex.num1)

# Summary statistics
summary(alex.num1)

# Distribution shape
skewness(alex.num1) # Total_Drivers, Serious_Injuries, Fatalities are skewed
hist(alex.num1$TotalDrivers)
hist(alex.num1$SeriousInjuries)
hist(alex.num1$Fatalities)

# Outliers
kurtosis(alex.num1) # quantitative check: 
#                    Total_Drivers, Serious_Injuries, Fatalities have outliers
boxplot(alex.num1[c('TotalDrivers','SeriousInjuries','Fatalities')]) # visual check/confirmation

# boxcox: Total_Drivers, Serious_Injuries, Fatalities have outliers
## Add small positive value to all zeros in numeric data frame
alex.num1[alex.num1 == 0] <- 0.00001

## DriverAge transformation
## Lambda search method 1: boxcox()
boxcox(alex.num1$DriverAge, optimize = T, lambda = c(-5,2)) # optimal lambda = 0.7785244

## lambda search method 2: BoxCox.lambda()
BoxCox.lambda(alex.num1$DriverAge) # optimal lambda = 0.6440091

b <- (alex.num1$DriverAge^0.6440091 - 1/0.6440091) # went with BoxCox.lambda() result
alex.num1$TransDriverAge <- b # add transformed data to data frame
alex.num1 <- subset(alex.num1, select = -c(DriverAge)) # drop untransformed feature
glimpse(alex.num1)
head(alex.num1)
boxplot(alex.num1$TransDriverAge)

# ______________________________ CORRELATION ___________________________________

# Multicollinearty check
A <- lm(SeriousInjuries ~., data = alex.num1)
vif(A) # no multicollinearity issues

cor(alex.num1) # 

# __________________________ FE: Single target features ________________________

glimpse(alex.num)
# Make 'Serious_Injuries' and 'Fatalities' the only target variable in separate
# data frames
serious.alex <- subset(alex, select = -c(Fatalities, DriverAge))
serious.alex <- mutate_if(serious.alex, is.character, as.factor)
str(serious.alex) # sanity check

## Add transformed driver age and serious injuries to serious.alex
serious.alex$TransDriverAge <- alex.num1$TransDriverAge
glimpse(serious.alex)

fatal.alex <- subset(alex, select = -c(SeriousInjuries, DriverAge))
fatal.alex <- mutate_if(fatal.alex, is.character, as.factor)
glimpse(fatal.alex) # sanity check
str(fatal.alex)

## Add transformed driver age and fatalities to fatal.alex
fatal.alex$TransDriverAge <- alex.num1$TransDriverAge
glimpse(fatal.alex) # sanity check

# Sum up target feature values per month
## Serious injuries
glimpse(serious.alex)
serious.alex$CrashYear <- floor_date(serious.alex$CrashYear, 
                                        "month")
serious.alex <- serious.alex %>%                        
  group_by(CrashYear) %>% 
  dplyr::summarize(SeriousInjuries = sum(SeriousInjuries)) %>% 
  as.data.frame()
head(serious.alex)

## Fatalities
glimpse(fatal.alex)
fatal.alex$CrashYear <- floor_date(fatal.alex$CrashYear, 
                                     "month")
fatal.alex <- fatal.alex %>%                        
  group_by(CrashYear) %>% 
  dplyr::summarize(Fatalities = sum(Fatalities)) %>% 
  as.data.frame()
head(fatal.alex)

# ______________________ FEATURE SELECTION (FS) ________________________________

# Feature selection with decision trees

# Serious_Injuries
serious.tree1 <- rpart(SeriousInjuries~.-Fatalities-DriverAge-CrashYear, data = alex, control = rpart.control(cp = 0.01))
summary(serious.tree1)
# cp = 0.01; MSE = 0.3049142; rel error = 0.4972371
barplot(serious.tree1$variable.importance)

serious.tree2 <- rpart(SeriousInjuries~.-Fatalities-DriverAge-CrashYear, data = alex, control = rpart.control(cp = 0.025))
summary(serious.tree2)
# cp = 0.025; MSE = 0.3049142; rel error = 0.5922421

# Fatalities
fatal.tree1 <- rpart(Fatalities ~.-SeriousInjuries-DriverAge-CrashYear, data = alex, control = rpart.control(cp = 0.01))
summary(fatal.tree1)
# cp = 0.01; MSE = 0.125547; rel error = 0.4269144  
barplot(fatal.tree1$variable.importance)

fatal.tree2 <- rpart(Fatalities ~.-SeriousInjuries-DriverAge-CrashYear, data = alex, control = rpart.control(cp = 0.025))
summary(fatal.tree2)
# cp = 0.025; MSE = 0.125547; rel error = 0.4564702  
barplot(fatal.tree2$variable.importance)

# _______________________ Train-Test split _____________________________________
# Train/test splits: 70% train; 30% test
## createDataPartition() samples from inside factor levels as well
serious.train.idx <- createDataPartition(serious.alex$SeriousInjuries, p = 0.7, list = F)
serious.train <- serious.alex[serious.train.idx,]
serious.test <- serious.alex[-serious.train.idx,]

fatal.train.idx <- createDataPartition(fatal.alex$Fatalities, p = 0.7, list = F)
fatal.train <- fatal.alex[fatal.train.idx,]
fatal.test <- fatal.alex[-fatal.train.idx,]

# MODELING --------------------------------------

# Model 1: linear model ________________________________________________________
## _________________________ Serious injuries __________________________________
glimpse(serious.alex) # sanity check

serious.linear <- lm(SeriousInjuries ~., data = serious.train)
summary(serious.linear)

## Diagnostics analysis
# Normality check
par(mfrow = c(1,2), oma = c(0,0,2,0))
qqnorm(serious.linear$residuals)
qqline(serious.linear$residuals, col = 'red')

# Constant variance check
plot(fitted(serious.linear), serious.linear$residuals,
     xlab = 'Fitted values',
     ylab = 'Residuals',
     main = 'Residuals vs. Fitted Values')
abline(h = 0, col = 'blue')
mtext('Linear Model Diagnostics Analysis: Serious Injuries', outer = T, cex = 1.5)

## ___________________________ Fatalities ______________________________________
glimpse(fatal.alex)
fatal.linear <- lm(Fatalities ~., data = fatal.train)
summary(fatal.linear)

## Diagnostics analysis
# Normality check
par(mfrow = c(1,2), oma = c(0,0,2,0))
qqnorm(fatal.linear$residuals)
qqline(fatal.linear$residuals, col = 'red')

# Constant variance check
plot(fitted(fatal.linear), fatal.linear$residuals,
     xlab = 'Fitted values',
     ylab = 'Residuals',
     main = 'Residuals vs. Fitted Values')
abline(h = 0, col = 'blue')
mtext('Linear Model Diagnostics Analysis: Fatal Injuries', outer = T, cex = 1.5)

## Diagnostics show that:
## 1. residuals are not normally distributed
## 2. there isn't constant variance for
##    SeriousInjuries and Fatalities due to p-values lower than 0.05
## I conclude that a linear model will not be the optimal choice 

# Model 2: Multivariate Adaptive Regression Splines (MARS) _____________________
# SeriousInjuries
serious.MARS <- earth(SeriousInjuries ~., data = serious.train)
summary(serious.MARS)
serious.MARS

serious.MARS.train <- predict(serious.MARS, serious.train)
serious.MARS.train.rmse <- RMSE(serious.MARS.train, serious.train$SeriousInjuries) 
serious.MARS.train.rmse # 2.338538

serious.MARS.test <- predict(serious.MARS, serious.test)
serious.MARS.test.rmse <- RMSE(serious.MARS.test, serious.test$SeriousInjuries) 
serious.MARS.test.rmse # 2.374308

# Fatalities
fatal.MARS <- earth(Fatalities ~., data = fatal.train)
summary(fatal.MARS)

fatal.MARS.train <- predict(fatal.MARS, fatal.train)
fatal.MARS.train.rmse <- RMSE(fatal.MARS.train, fatal.train$Fatalities) 
fatal.MARS.train.rmse # 0.8017837

fatal.MARS.test <- predict(fatal.MARS, fatal.test)
fatal.MARS.test.rmse <- RMSE(fatal.MARS.test, fatal.test$Fatalities) 
fatal.MARS.test.rmse # 0.5838742

# Model 3: Support Vector Machines (SVM) _______________________________________
# Kernel: Polynomial SVM ______________________________________________________

glimpse(serious.alex)

# SeriousInjuries
serious.svm1 <- svm(SeriousInjuries ~., data = serious.train, kernel = 'polynomial', cost = 0.5, gamma = 0.57, degree = 3)

serious.svm1.train <- predict(serious.svm1, serious.train)
serious.svm1.train.rmse <- RMSE( serious.svm1.train, serious.train$SeriousInjuries)
serious.svm1.train.rmse # 2.36617

serious.svm1.test <- predict(serious.svm1, serious.test)
serious.svm1.test.rmse <- RMSE(serious.svm1.test, serious.test$SeriousInjuries) 
serious.svm1.test.rmse # 2.426237

# Fatalities
fatal.svm1 <- svm(Fatalities ~., data = fatal.train, kernel = 'polynomial', cost = 0.5, gamma = 0.57, degree = 3)

fatal.svm1.train <- predict(fatal.svm1, fatal.train)
fatal.svm1.train.rmse <- RMSE(fatal.svm1.train, fatal.train$Fatalities) 
fatal.svm1.train.rmse # 0.9047092

fatal.svm1.test <- predict(fatal.svm1, fatal.test)
fatal.svm1.test.rmse <- RMSE(fatal.svm1.test, fatal.test$Fatalities) 
fatal.svm1.test.rmse # 0.6342331

# Kernel: Radial SVM __________________________________________________________

# SeriousInjuries
serious.svm2 <- svm(SeriousInjuries ~., data = serious.train, kernel = 'radial', gamma = 3.026316, cost = 15)

serious.svm2.train <- predict(serious.svm2, serious.train)
serious.svm2.train.rmse <- RMSE(serious.svm2.train, serious.train$SeriousInjuries) 
serious.svm2.train.rmse # 2.18647

serious.svm2.test <- predict(serious.svm2, serious.test)
serious.svm2.test.rmse <- RMSE(serious.svm2.test, serious.test$SeriousInjuries) 
serious.svm2.test.rmse # 2.205921

# Fatalities
fatal.svm2 <- svm(Fatalities ~., data = fatal.train, kernel = 'radial', gamma = 3.026316, cost = 15)

fatal.svm2.train <- predict(fatal.svm2, fatal.train)
fatal.svm2.train.rmse <- RMSE(fatal.svm2.train, fatal.train$Fatalities) 
fatal.svm2.train.rmse # 0.843663

fatal.svm2.test <- predict(fatal.svm2, fatal.test)
fatal.svm2.test.rmse <- RMSE(fatal.svm2.test, fatal.test$Fatalities)
fatal.svm2.test.rmse # 0.6695157

# Model 4: Random Forests ______________________________________________________

# SeriousInjuries
serious.rfGrid <- expand.grid(mtry = 2:9) 
serious.rf <- train(SeriousInjuries ~.,
                    data = serious.train,
                    method = 'rf', 
                    trControl = trainControl(method = 'cv', number = 5), 
                    tuneGrid = serious.rfGrid) 

serious.rf.train <- predict(serious.rf, serious.train)
serious.rf.train.rmse <- RMSE(serious.rf.train, serious.train$SeriousInjuries) 
serious.rf.train.rmse # 1.369182

serious.rf.test <- predict(serious.rf, serious.test)
serious.rf.test.rmse <- RMSE(serious.rf.test, serious.test$SeriousInjuries)
serious.rf.test.rmse # 2.247285

# Fatalities
fatal.rfGrid <- expand.grid(mtry = 2:9)
fatal.rf <- train(Fatalities ~.,
                    data = fatal.train,
                    method = 'rf', 
                    trControl = trainControl(method = 'cv', number = 5), 
                    tuneGrid = fatal.rfGrid)

fatal.rf.train <- predict(fatal.rf, fatal.train)
fatal.rf.train.rmse <- RMSE(fatal.rf.train, fatal.train$Fatalities)
fatal.rf.train.rmse # 0.5353803

fatal.rf.test <- predict(fatal.rf, fatal.test)
fatal.rf.test.rmse <- RMSE(fatal.rf.test, fatal.test$Fatalities) 
fatal.rf.test.rmse # 0.8114454

# Model 5: Neural Network ______________________________________________________
glimpse(serious.alex.train)
str(serious.alex.train)

# NN1: Neural net with one hidden layer (method = 'nnet')

# SeriousInjurires
serious.nn1.grid <- expand.grid(decay = seq(0.5, 0.1), size = seq(5, 6))
serious.nn1 <- train(SeriousInjuries ~.,
                     data = serious.train,
                     method = "nnet", 
                     metric = 'RMSE',
                     tuneGrid = serious.nn1.grid,
                     preProc = c('center','scale','nzv'),
                     trControl = trainControl(method = 'cv', number = 5, verboseIter = F))

serious.nn1.train <- predict(serious.nn1, serious.train)
serious.nn1.train.rmse <- RMSE(serious.nn1.train, serious.train$SeriousInjuries) 
serious.nn1.train.rmse # 3.498518

serious.nn1.test <- predict(serious.nn1, serious.test)
serious.nn1.test.rmse <- RMSE(serious.nn1.test, serious.test$SeriousInjuries)
serious.nn1.test.rmse # 3.642651

# Fatalities
fatal.nn1.grid <- expand.grid(decay = seq(0.5, 0.1), size = seq(5, 6))
fatal.nn1 <- train(Fatalities ~.,
                   data = fatal.train,
                   method = "nnet",
                   metric = 'RMSE',
                   tuneGrid = fatal.nn1.grid,
                   preProc = c('center','scale','nzv'),
                   trControl = trainControl(method = 'cv', number = 5, verboseIter = F))

fatal.nn1.train <- predict(fatal.nn1, fatal.train)
fatal.nn1.train.rmse <- RMSE(fatal.nn1.train, fatal.train$Fatalities) 
fatal.nn1.train.rmse # 0.8017837

fatal.nn1.test <- predict(fatal.nn1, fatal.test)
fatal.nn1.test.rmse <- RMSE(fatal.nn1.test, fatal.test$Fatalities)
fatal.nn1.test.rmse # 0.5838769

# NN2:  Multi-layer perceptron with weight decay (mlpWeightDecayML)

# SeriousInjuries
serious.nn2.grid <- expand.grid(layer1 = 10, layer2 = 10, 
                        layer3 = 10, decay = seq(1,1e-10))
serious.nn2 <- train(SeriousInjuries ~.,
                     data = serious.train,
                     method = "mlpWeightDecayML",
                     tuneGrid = serious.nn2.grid,
                     metric = 'RMSE',
                     preProc = c('center','scale','nzv'),
                     trControl = trainControl(method = 'cv', number = 5, verboseIter = F))

serious.nn2.train <- predict(serious.nn2, serious.train)
serious.nn2.train.rmse <- RMSE(serious.nn2.train, serious.train$SeriousInjuries) 
serious.nn2.train.rmse # 2.440662

serious.nn2.test <- predict(serious.nn2, serious.test)
serious.nn2.test.rmse <- RMSE(serious.nn2.test, serious.test$SeriousInjuries) 
serious.nn2.test.rmse # 2.520752

# Fatalities
fatal.nn2.grid <- expand.grid(layer1 = 10, layer2 = 10, 
                                layer3 = 10, decay = seq(1,1e-10))
fatal.nn2 <- train(Fatalities ~.,
                     data = fatal.train,
                     method = "mlpWeightDecayML",
                     tuneGrid = fatal.nn2.grid,
                     metric = 'RMSE',
                     preProc = c('center','scale','nzv'),
                     trControl = trainControl(method = 'cv', number = 5, verboseIter = F))

fatal.nn2.train <- predict(fatal.nn2, fatal.train)
fatal.nn2.train.rmse <- RMSE(fatal.nn2.train, fatal.train$Fatalities)
fatal.nn2.train.rmse # 0.7916862

fatal.nn2.test <- predict(fatal.nn2, fatal.test)
fatal.nn2.test.rmse <- RMSE(fatal.nn2.test, fatal.test$Fatalities)
fatal.nn2.test.rmse # 0.5683812


# Model 6: ARIMA _______________________________________________________________

# SeriousInjuries 
glimpse(serious.alex) # sanity check
tail(serious.alex)
tail(fatal.alex)

serious.arima <- auto.arima(serious.train$SeriousInjuries)
summary(serious.arima)

nrow(serious.train) # 57 rows in train set
nrow(serious.test) # 21 rows in test set

serious.arima.train <- predict(serious.arima, 57) # 57 rows in train set (serious.train)
serious.arima.train.rmse <- RMSE(serious.arima.train$pred, serious.train$SeriousInjuries)
serious.arima.train.rmse # RMSE = 2.338538

serious.arima.test <- predict(serious.arima, 21) # 21 rows in test set (serious.test)
serious.arima.test.rmse <- RMSE(serious.arima.test$pred, serious.test$SeriousInjuries)
serious.arima.test.rmse # RMSE = 2.374308

# Fatalities 
glimpse(fatal.alex) # sanity check

fatal.arima <- auto.arima(fatal.train$Fatalities)
summary(fatal.arima)

nrow(fatal.train) # 56 rows in train set
nrow(fatal.test) # 22 rows in test set

fatal.arima.train <- predict(fatal.arima, 56) # 56 rows in train set (fatal.train)
fatal.arima.train.rmse <- RMSE(fatal.arima.train$pred, fatal.train$Fatalities)
fatal.arima.train.rmse # RMSE = 0.8017837

fatal.arima.test <- predict(fatal.arima, 22) # 22 rows in test set (serious.test)
fatal.arima.test.rmse <- RMSE(fatal.arima.test$pred, fatal.test$Fatalities)
fatal.arima.test.rmse # RMSE = 0.5838742

# MODEL EVALUATION ---------------------------------


## Serious injuries models
serious.finalists <- data.frame("serious.MARS" = c(serious.MARS.train.rmse, serious.MARS.test.rmse),
                                'serious.SVM1' = c(serious.svm1.train.rmse,serious.svm1.test.rmse),
                                'serious.SVM2' = c(serious.svm2.train.rmse,serious.svm2.test.rmse),
                                'serous.RF' = c(serious.rf.train.rmse, serious.rf.test.rmse),
                                'serious NN1' = c(serious.nn1.train.rmse, serious.nn1.test.rmse),
                                'serious NN2' = c(serious.nn2.train.rmse, serious.nn2.test.rmse),
                                'serious ARIMA' = c(serious.arima.train.rmse, serious.arima.test.rmse),
                                row.names = c('Train RMSE','Test RMSE'))
serious.finalists # sanity check

## Fatalities models
fatal.finalists <- data.frame("fatal.MARS" = c(fatal.MARS.train.rmse, fatal.MARS.test.rmse),
                              'fatal.SVM1' = c(fatal.svm1.train.rmse, fatal.svm1.test.rmse),
                              'fatal.SVM2' = c(fatal.svm2.train.rmse, fatal.svm2.test.rmse),
                              'fatal RF' = c(fatal.rf.train.rmse, fatal.rf.test.rmse),
                              'fatal NN1' = c(fatal.nn1.train.rmse, fatal.nn1.test.rmse),
                              'fatal NN2' = c(fatal.nn2.train.rmse, fatal.nn2.test.rmse),
                              'fatal ARIMA' = c(fatal.arima.train.rmse, fatal.arima.test.rmse),
                              row.names = c('Train RMSE','Test RMSE'))
fatal.finalists # sanity check

# Transpose data frames
serious.finalists1 <- data.frame(t(serious.finalists))
serious.finalists1

fatal.finalists1 <- data.frame(t(fatal.finalists))
fatal.finalists1

# Calculate difference between test and train RMSE, then add differences as column.

## My logic here is to quantitatively understand the extent of overfitting.
## Large train-test values = bad; small train-test values = better.
serious.finalists1$Train.Test.Difference <- serious.finalists1$Test.RMSE - serious.finalists1$Train.RMSE
serious.finalists1

fatal.finalists1$Train.Test.Difference <- fatal.finalists1$Test.RMSE - fatal.finalists1$Train.RMSE
fatal.finalists1

# Sort finalists by lowest RMSE then lowest train test difference
serious.finalists1 <- serious.finalists1[order(serious.finalists1$Test.RMSE,
                         serious.finalists1$Train.Test.Difference),]
serious.finalists1 # sanity check

fatal.finalists1 <- fatal.finalists1[order(fatal.finalists1$Test.RMSE,
                       fatal.finalists1$Train.Test.Difference),]
fatal.finalists1 # sanity check

# Conclusion:
# For serious injuries I will use the second support vector machine model (serious.SVM2)
# For fatalities I will use the second neural network model (fatal.NN2)
# So the first neural network (NN1) will be used to predict serious injuries

# DEPLOYMENT ---------------------------------------
serious.model <- serious.svm2
fatal.model <- fatal.nn2
serious.model
fatal.model

dim(serious.predictions.df)
plot(forecast(fatal.model))
# Create start and end dates
start.date <- ymd('2022/01/01')
end.date <- ymd('2029/01/01')

# Predict serious injuries for 2022-2029
serious.2028 <- data.frame(CrashYear = seq(start.date, end.date,'months'))

serious.predictions <- predict(serious.model, serious.2028)
serious.predictions <- floor(0.5 + serious.predictions) # round up numbers with decimals >= 0.5
serious.predictions # sanity check

# Predict fatalities for 2022-2029
fatal.2028 <- data.frame(CrashYear = seq(start.date, end.date,'months'))
fatal.2028

fatal.predictions <- predict(fatal.model, fatal.2028)
fatal.predictions <- floor(0.5 + fatal.predictions) # round up numbers with decimals >= 0.5
fatal.predictions # sanity check

# Merge future dates with predictions
serious.predictions.df <- cbind(serious.2028, serious.predictions)
serious.predictions.df$SeriousInjuries <- serious.predictions.df$serious.predictions
head(serious.predictions.df)
serious.predictions.df <- subset(serious.predictions.df, select = -c(serious.predictions))
head(serious.predictions.df)

fatal.predictions.df <- cbind(fatal.2028, fatal.predictions)
fatal.predictions.df$Fatalities <- fatal.predictions.df$fatal.predictions
head(fatal.predictions.df)
fatal.predictions.df <- subset(fatal.predictions.df, select = -c(fatal.predictions))
head(fatal.predictions.df)

# _________________________ VISUALISATIONS _____________________________________

tail(serious.alex)
serious.alex <- serious.alex[-c(78),]

tail(fatal.alex)
fatal.alex <- fatal.alex[-c(78),]

serious.22.28 <- rbind(serious.alex, serious.predictions.df)
fatal.22.28 <- rbind(fatal.alex, fatal.predictions.df)

# Serious injuries
serious.plot <- ggplot(serious.22.28, aes(x = CrashYear, y = SeriousInjuries)) +
  geom_line() + scale_x_date(date_breaks = "1 year", date_labels = '%Y') +
  scale_y_continuous(breaks = seq(1, 13, 1)) +
  ggtitle('Serious Injuries Prediction for the City of Alexandria') +
  labs(x = 'Crash Year', y = 'Number of Serious Injuries') +
  geom_vline(xintercept = as.numeric(as.Date(c("2028-01-01","2029-01-01"))), linetype=2) +
  geom_rect(aes(xmin = as.Date("2028-01-01"),
                xmax = as.Date("2029-01-01"),
                ymin = -Inf, ymax = Inf), fill = "orange", alpha = 0.008)
serious.plot

# Fatalities
fatal.plot <- ggplot(fatal.22.28, aes(x = CrashYear, y = Fatalities)) +
  geom_line() + scale_x_date(date_breaks = "1 year", date_labels = '%Y') +
  scale_y_continuous(breaks = seq(1, 13, 1)) +
  ggtitle('Fatalities Prediction for the City of Alexandria') +
  labs(x = 'Crash Year', y = 'Number of Fatalities') +
  geom_vline(xintercept = as.numeric(as.Date(c("2028-01-01","2029-01-01"))), linetype=2) +
  geom_rect(aes(xmin = as.Date("2028-01-01"),
                xmax = as.Date("2029-01-01"),
                ymin = -Inf, ymax = Inf), fill = "red", alpha = 0.008)
fatal.plot
