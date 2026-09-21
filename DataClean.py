# import pandas as pd
from data_process_lib import *


cleaning_feedback = False
vectoriser_feedback = True

# Import data
df = pd.read_csv('BCIT601_starter_service_requests_dataset.csv')
# clean data
df = clean_training_data(df,cleaning_feedback)

# Prep training data
x_train, x_test, y_train, y_test = split_training_data(df)
#print(x_train)

# Vectorise data
x_train_vec, x_test_vec = vectorize_text(x_train, x_test, vectoriser_feedback)

# export_cols_to_csv("training_data.csv", x_train,y_train)



# Export cleaned data
# df.to_csv('clean_data.csv', index=False)