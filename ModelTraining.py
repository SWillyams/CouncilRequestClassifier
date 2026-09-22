# import pandas as pd
from data_process_lib import *


cleaning_feedback = False
vectoriser_feedback = False
export_training_data = False
model_type = 'any'
build_model_feedback = True

# Import data
df = pd.read_csv('BCIT601_starter_service_requests_dataset.csv')
# clean data
df = clean_training_data(df,cleaning_feedback)

# Prep training data
x_train, x_test, y_train, y_test = split_training_data(df)
#print(x_train)

# Vectorise data
x_train_vec, x_test_vec, feature_list = vectorize_text(x_train, x_test, vectoriser_feedback)

if export_training_data:
    export_cols_to_csv("training_data.csv", feature_list,x_train_vec, x_train,y_train)
del feature_list

#Build the models
if model_type == 'any' or model_type == 'log_reg':
    build_log_regres(x_train_vec,x_test_vec,y_train,y_test,build_model_feedback)
if model_type == 'any' or model_type == 'multi_nb':
    build_multinomial_nb(x_train_vec,x_test_vec,y_train,y_test,build_model_feedback)


# Export cleaned data
# df.to_csv('clean_data.csv', index=False)