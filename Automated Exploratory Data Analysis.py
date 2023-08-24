#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[53]:


# Load the data
data = pd.read_csv(r"E:\International_Report_Departures.csv")


# In[54]:


data.head()


# In[56]:


data=data.iloc[:5,:5]


# In[57]:


data


# In[58]:


data=data.to_csv("subset_data.csv")


# In[59]:


data= pd.read_csv("subset_data.csv")


# In[60]:


data


# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    
    imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    
    data[non_numeric_cols] = data[non_numeric_cols].fillna('Unknown')
    
    encoder = OneHotEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = encoder.fit_transform(data[categorical_cols])
    
    # You can use feature names provided by encoder or manually generate them
    feature_names_out = encoder.get_feature_names_out(input_features=categorical_cols)
    data_encoded_df = pd.DataFrame(data_encoded.toarray(), columns=feature_names_out)
    
    preprocessed_data = pd.concat([data[numeric_cols], data_encoded_df], axis=1)
    return preprocessed_data


def create_visualizations(data, column_name, column_type):
    if column_type == 'numerical':
        sns.histplot(data[column_name])
        plt.title(f'Histogram of {column_name}')
        plt.show()
        
    elif column_type == 'categorical':
        sns.countplot(data[column_name])
        plt.title(f'Count Plot of {column_name}')
        plt.show()

def main():
    print("Welcome to Automated EDA Tool!")

    data = pd.read_csv("subset_data.csv")
    
    preprocessed_data = preprocess_data(data)
    
    numeric_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
    categorical_cols = preprocessed_data.select_dtypes(include=['object']).columns
    
    for column in numeric_cols:
        create_visualizations(preprocessed_data, column, 'numerical')
    
    for column in categorical_cols:
        create_visualizations(preprocessed_data, column, 'categorical')
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()


# In[ ]:




