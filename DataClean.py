# import pandas as pd
from data_process_lib import *

# Import data
df = pd.read_csv('BCIT601_starter_service_requests_dataset.csv')

#Remove excess columns before testing for duplicates
df = df.drop(columns=['request_id','priority','date_reported'])
#Remove duplicates
df.remove_duplicates()

print(df)
# force lowercase

""" def lower_array(array):
    return [request.lower() for item in array]
lowered_requests = [request.lower() for request in requests]
lowered_requests = lower_array(requests) """

# remove duplicates


# df.to_csv('clean_data.csv', index=False)

# remove punctuation

# remove stop words

# Export cleaned data