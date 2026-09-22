# import pandas as pd
from data_process_lib import *


cleaning_feedback = False
vectoriser_feedback = False
export_training_data = False
model_type = 'log_reg'
test_model_feedback = False

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
if test_model_feedback:
    test_text = x_test.reset_index(drop=True)
    test_true_cat = y_test.reset_index(drop=True)


if model_type == 'any' or model_type == 'log_reg':
    log_preds, _ = test_log_regres(x_train_vec,x_test_vec,y_train,y_test,test_model_feedback)

    if test_model_feedback:
        log_reg_result = pd.DataFrame({
            "Request_text" : x_test.reset_index(drop=True),
            "True category" : y_test.reset_index(drop=True),
            "Predicted Category" : log_preds
        })
        print(log_reg_result.to_string(index=False))


if model_type == 'any' or model_type == 'multi_nb':
    nb_preds, _ = test_multinomial_nb(x_train_vec,x_test_vec,y_train,y_test,test_model_feedback)

    if test_model_feedback:
        log_reg_result = pd.DataFrame({
            "Request_text" : x_test.reset_index(drop=True),
            "True category" : y_test.reset_index(drop=True),
            "Predicted Category" : nb_preds
        })
        print(log_reg_result.to_string(index=False))


random_states = [42,1,2,3,4,5,6,7,8,9]
# test_models_repeated(df, model_type,random_states)

# Export cleaned data
# df.to_csv('clean_data.csv', index=False)