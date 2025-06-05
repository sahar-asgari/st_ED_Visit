#%%
# import libraries
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static

import matplotlib.pyplot as plt
from datetime import datetime
#%%
today = datetime.now()
today_day = today.day
today_month = today.month
today_year = today.year
#%%
#To get the most recent institutions use last one year of ED data
New_ED = pd.read_csv('New_ED_data.csv')
New_ED['date'] = pd.to_datetime(New_ED['date'], format='mixed')
unique_inst_no = New_ED['inst_no'].unique()

#To get hospitals information such as lat and long for the map
hospital_df = pd.read_csv('INST_LAT_LONG.csv', encoding='latin-1')
hospital_df.columns = hospital_df.columns.str.lower()
hospital_df = hospital_df[hospital_df['inst_no'].isin(unique_inst_no)]

#Dashboard table starts from 2022
Dashboard_start_date = '2022-01-01'
Last_ED_available_date = '2024-07-01'
latest_year_availableED = 2024
latest_month_availableED = 6
# Calculate the average daily ED visits for each season for last 2 years
start_date = '2022-06-01'
end_date = '2024-06-30'

#%%
##################################################################################################################################
##################################################################################################################################
#data preprocessing 
@st.cache_resource
def load_data():
    ED_concat = pd.read_csv('ED_concat.csv')
    ED_concat['date'] = pd.to_datetime(ED_concat['date'], format='mixed')
    inst_hosp = pd.read_csv('Inst_Hosp.csv', encoding='latin-1') #, dtype=dtype_dict)
    return ED_concat, inst_hosp

ED_concat, inst_hosp = load_data()
inst_numbers = [int(inst_no) for inst_no in unique_inst_no]
inst_no_name_map = dict(zip(inst_hosp['inst_no'], inst_hosp['inst_name']))
#%%
@st.cache_resource
def data_preprocess():
    #Data preprocess
    df = ED_concat.copy()

    # Now you can extract year, month, and day
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    #Dashboard table starts from 2022
    df = df[df['date'] >= Dashboard_start_date]

    # Filter out the rows for the specific institution and date range because ED visits freq for these inst are missed in NACRS
    df = df[~((df['inst_no'] == 4109) & (df['date'] >= '2023-10-01') & (df['date'] <= '2024-03-31'))]
    df = df[~((df['inst_no'] == 4166) & (df['date'] >= '2023-04-01') & (df['date'] <= '2024-04-30'))]
    df = df[~((df['inst_no'] == 4250) & (df['date'] >= '2022-09-01') & (df['date'] <= '2023-03-31'))]
    df = df[~((df['inst_no'] == 4256) & (df['date'] >= '2023-10-01') & (df['date'] <= '2024-03-31'))]
   
    df = df.reset_index(drop=True)
    df['smooth_ed_visits'] = df['total_ed']

    columns_to_shift = [
                        'max_Co', 'min_Co', 'mean_Co',
                        'max_O3', 'min_O3', 'mean_O3',
                        'max_FPM', 'min_FPM', 'mean_FPM',
                        'max_No2', 'min_No2', 'mean_No2',
                        'max_So2', 'min_So2', 'mean_So2']
    for lag in [3, 7]:
        for var in columns_to_shift:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

    def prepare_dataset(lag):
        selected_cols = ['year', 'month', 'day', 'inst_no', 'pandemic', 'max_t', 'min_t', 'precipitation', 'smooth_ed_visits']
        selected_cols += [f'{var}_lag{lag}' for var in columns_to_shift]
        df_lag = df[selected_cols]

        #includes data from 2022 to today
        prediction_set = df_lag.fillna(0)
        X_prediction = prediction_set.drop(columns=['smooth_ed_visits'])

        df_lag['date'] = pd.to_datetime(df_lag[['year', 'month', 'day']])
        #includes data from 2022 to Last_ED_available_date
        df_train = df_lag[df_lag['date'] < Last_ED_available_date]
        return X_prediction, df_train, df_lag
    
    X_prediction_3, df_train_3, df_lag3 = prepare_dataset(3)
    X_prediction_7, df_train_7, df_lag7 = prepare_dataset(7)

    return X_prediction_3, X_prediction_7, df_train_3, df_train_7, df_lag3, df_lag7

X_prediction_3, X_prediction_7, df_train_3, df_train_7, df_lag3, df_lag7 = data_preprocess()
##################################################################################################################################
##################################################################################################################################
#%%
#Machine Learning - Function to load models with caching
@st.cache_resource
def MLs():
    with open('RF_reg_3.pkl', 'rb') as f:
        RF_reg_3 = pickle.load(f)
        print("RF_reg_3 Model loaded successfully!")

    with open('RF_reg_7.pkl', 'rb') as f:
        RF_reg_7 = pickle.load(f)
        print("RF_reg_7 Model loaded successfully!")
    return RF_reg_3, RF_reg_7
RF_reg_3, RF_reg_7 = MLs()
#%%
@st.cache_resource
def make_predictions(_RF_reg_3, _RF_reg_7, X_prediction_3, X_prediction_7):
    def predict_and_prepare(model, X_prediction):
        y_prediction = model.predict(X_prediction).round()
        df_predict = X_prediction.copy()
        df_predict['Predicted_ED_Visits'] = y_prediction        
        return df_predict

    df_predict_3 = predict_and_prepare(RF_reg_3, X_prediction_3)
    df_predict_7 = predict_and_prepare(RF_reg_7, X_prediction_7)
    
    return df_predict_3, df_predict_7

df_predict_3, df_predict_7 = make_predictions(RF_reg_3, RF_reg_7, X_prediction_3, X_prediction_7)
df_predict_3['smooth_ed_visits'] = df_train_3['smooth_ed_visits'].copy()
df_predict_7['smooth_ed_visits'] = df_train_7['smooth_ed_visits'].copy()
#%%
##################################################################################################################################
##################################################################################################################################
@st.cache_resource
def seasonal_effect():
    # Define a function to determine the season based on month and day
    def get_season(month):
        if 3 <= month <= 5:
            return 'Spring'
        elif 6 <= month <= 8:
            return 'Summer'
        elif 9 <= month <= 11:
            return 'Fall'
        else:
            return 'Winter'
        
    # Apply the function to create a new 'Season' column for both dataframes
    for df in [df_predict_3, df_predict_7]:
        df['Season'] = df['month'].apply(get_season)

    # Calculate the average daily ED visits for each season for last 2 years
    season_avg_df_3 = df_predict_3.copy()
    season_avg_df_3['date'] = pd.to_datetime(season_avg_df_3[['year', 'month', 'day']])
    season_avg_df_3 = season_avg_df_3.loc[(season_avg_df_3['date'] >= start_date) & (season_avg_df_3['date'] <= end_date)]
    seasonal_avg_ED_visits_3 = season_avg_df_3.groupby(['Season', 'inst_no'])['smooth_ed_visits'].mean().round()
    
    season_avg_df_7 = df_predict_7.copy()
    season_avg_df_7['date'] = pd.to_datetime(season_avg_df_7[['year', 'month', 'day']])
    season_avg_df_7 = season_avg_df_7.loc[(season_avg_df_7['date'] >= start_date) & (season_avg_df_7['date'] <= end_date)]
    seasonal_avg_ED_visits_7 = season_avg_df_7.groupby(['Season', 'inst_no'])['smooth_ed_visits'].mean().round()

    def get_seasonal_average(month, inst_no, model_dropdown):
        # Adjust seasonal_avg_ED_visits based on the selected model
        if model_dropdown == 'ED Visits after 3 Days':
            seasonal_avg_ED_visits = seasonal_avg_ED_visits_3
        else:
            seasonal_avg_ED_visits = seasonal_avg_ED_visits_7


        season = get_season(month)
        return seasonal_avg_ED_visits.get((season, inst_no), "No data")

    return get_seasonal_average

get_seasonal_average = seasonal_effect()
#%%
##################################################################################################################################
##################################################################################################################################
@st.cache_resource
def get_historical_data(year, month, day, inst_no, model_dropdown):
    # Adjust lag based on the selected model
    if model_dropdown == 'ED Visits after 3 Days':
        df_predict = df_predict_3
    else:
        df_predict = df_predict_7
    
    # Filter data for specific date and institution
    filtered_data = df_predict[(df_predict['year'] == year) & (df_predict['month'] == month) & (df_predict['day'] == day) & (df_predict['inst_no'] == inst_no)]

    # Check if data exists
    if filtered_data.empty:
        return None

    # Adjust lag based on the selected model
    if model_dropdown == 'ED Visits after 3 Days':
        lag_column = 'lag3'
    else:
        lag_column = 'lag7'
    
    # Extract relevant data
    weather_vars = ['pandemic', 'max_t', 'min_t', 'precipitation']
    air_quality_vars = ['max_Co', 'min_Co', 'mean_Co', 'max_O3', 'min_O3', 'mean_O3', 
                        'max_FPM', 'min_FPM', 'mean_FPM', 'max_No2', 'min_No2', 'mean_No2', 
                        'max_So2', 'min_So2', 'mean_So2']

    historical_data = {var: filtered_data[var].values[0] for var in weather_vars}
    historical_data.update({
        f'{var}_lag': filtered_data[f'{var}_{lag_column}'].values[0] for var in air_quality_vars
    })

    return historical_data
#%%
##################################################################################################################################
##################################################################################################################################
# Month mapping
months = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
days = list(range(1, 32))  # Days from 1 to 31
years = list(range(2023, 2026))  # Years from 2011 to 2024

def create_date_streamlit():
    # initialization 
    if 'year_input' not in st.session_state:
        st.session_state.year_input = 2025
    if 'month_input' not in st.session_state:
        st.session_state.month_input = 1 
    if 'day_input' not in st.session_state:
        st.session_state.day_input = 1
    if 'inst_input' not in st.session_state:
        st.session_state.inst_input = inst_numbers[0]

    # Input widgets
    selected_year = st.selectbox('Year', options=years)
    st.session_state.year_input = selected_year  #update session state

    selected_month = st.selectbox('Month', options=list(months.keys()), format_func=lambda x: f"{x} - {months[x]}")
    st.session_state.month_input = selected_month
    

    selected_day = st.selectbox('Day', options=days)
    st.session_state.day_input = selected_day

    # Create a dropdown with both inst_no and hospital_name
    inst_options = {f"{inst_no} - {inst_no_name_map[inst_no]}": inst_no for inst_no in inst_numbers}
    # Sorting the dictionary by the hospital names
    inst_options = {k: v for k, v in sorted(inst_options.items(), key=lambda item: item[0].split(' - ')[1])}

    selected_inst = st.selectbox('Institution Number and Name', options=list(inst_options.keys()))
    st.session_state.inst_input = inst_options[selected_inst]

    return st.session_state.year_input, st.session_state.month_input, st.session_state.day_input, st.session_state.inst_input
#%%
# ##################################################################################################################################
##################################################################################################################################
def create_pandemic_streamlit():
    if 'pandemic_input' not in st.session_state:
        st.session_state.pandemic_input = 0

    st.session_state.pandemic_input = st.radio('Pandemic:', options=["No", "Yes"], index=0 if st.session_state.get('pandemic_input', 0) == 0 else 1)
    return st.session_state.pandemic_input

def create_weather_streamlit():
    st.session_state.max_temperature_input = st.slider('Max Temperature:', min_value=-48.0, max_value=39.0, value=st.session_state.get('max_temperature_input', 39.0))
    st.session_state.min_temperature_input = st.slider('Min Temperature:', min_value=-48.0, max_value=39.0, value=st.session_state.get('min_temperature_input', 27.0))
    st.session_state.precipitation_input = st.slider('Precipitation:', min_value=0.0, max_value=110.0, value=st.session_state.get('precipitation_input', 0.0))
    return st.session_state.max_temperature_input, st.session_state.min_temperature_input, st.session_state.precipitation_input

def create_air_quality_streamlit():
    st.session_state.max_Co_input = st.slider('Max CO:', min_value=0.0, max_value=4.0, value=st.session_state.get('max_Co_input', 0.0))
    st.session_state.min_Co_input = st.slider('Min CO:', min_value=0.0, max_value=4.0, value=st.session_state.get('min_Co_input', 0.0))
    st.session_state.mean_Co_input = st.slider('Mean CO:', min_value=0.0, max_value=4.0, value=st.session_state.get('mean_Co_input', 0.0))

    st.session_state.max_O3_input = st.slider('Max O3:', min_value=0.0, max_value=130.0, value=st.session_state.get('max_O3_input', 0.0))
    st.session_state.min_O3_input = st.slider('Min O3:', min_value=0.0, max_value=130.0, value=st.session_state.get('min_O3_input', 0.0))
    st.session_state.mean_O3_input = st.slider('Mean O3:', min_value=0.0, max_value=130.0, value=st.session_state.get('mean_O3_input', 0.0))

    st.session_state.max_FPM_input = st.slider('Max FPM:', min_value=0.0, max_value=122.0, value=st.session_state.get('max_FPM_input', 0.0))
    st.session_state.min_FPM_input = st.slider('Min FPM:', min_value=0.0, max_value=122.0, value=st.session_state.get('min_FPM_input', 0.0))
    st.session_state.mean_FPM_input = st.slider('Mean FPM:', min_value=0.0, max_value=122.0, value=st.session_state.get('mean_FPM_input', 0.0))

    st.session_state.max_No2_input = st.slider('Max NO2:', min_value=0.0, max_value=57.0, value=st.session_state.get('max_No2_input', 0.0))
    st.session_state.min_No2_input = st.slider('Min NO2:', min_value=0.0, max_value=57.0, value=st.session_state.get('min_No2_input', 0.0))
    st.session_state.mean_No2_input = st.slider('Mean NO2:', min_value=0.0, max_value=57.0, value=st.session_state.get('mean_No2_input', 0.0))

    st.session_state.max_So2_input = st.slider('Max SO2:', min_value=0.0, max_value=11.0, value=st.session_state.get('max_So2_input', 0.0))
    st.session_state.min_So2_input = st.slider('Min SO2:', min_value=0.0, max_value=11.0, value=st.session_state.get('min_So2_input', 0.0))
    st.session_state.mean_So2_input = st.slider('Mean SO2:', min_value=0.0, max_value=11.0, value=st.session_state.get('mean_So2_input', 0.0))

    return (
        st.session_state.max_Co_input, st.session_state.min_Co_input, st.session_state.mean_Co_input,
        st.session_state.max_O3_input, st.session_state.min_O3_input, st.session_state.mean_O3_input,
        st.session_state.max_FPM_input, st.session_state.min_FPM_input, st.session_state.mean_FPM_input,
        st.session_state.max_No2_input, st.session_state.min_No2_input, st.session_state.mean_No2_input,
        st.session_state.max_So2_input, st.session_state.min_So2_input, st.session_state.mean_So2_input
    )
##################################################################################################################################
##################################################################################################################################
def create_model_dropdown():
    model_options = {'ED Visits after 3 Days': RF_reg_3,'ED Visits after 7 Days': RF_reg_7}
    model_dropdown = st.selectbox('Select Model', options=model_options)  
    #Add gray border to the dropdown
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] > div {
            border: 2px solid gray;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    return model_dropdown

#If an instance exists in the df_predict data for a given date
def validate_inst_no(year, month, day, inst_no, model_dropdown):
    if model_dropdown == 'ED Visits after 3 Days':
        exists = not df_predict_3[(df_predict_3['year'] == year) & 
                                             (df_predict_3['month'] == month) & 
                                             (df_predict_3['day'] == day) & 
                                             (df_predict_3['inst_no'] == inst_no)].empty
        
    else:
        exists = not df_predict_7[(df_predict_7['year'] == year) & 
                                             (df_predict_7['month'] == month) & 
                                             (df_predict_7['day'] == day) & 
                                             (df_predict_7['inst_no'] == inst_no)].empty
    return exists
##################################################################################################################################
##################################################################################################################################
def predict_ED_visits(model_dropdown):
    # Get input values
    year = st.session_state.year_input
    month = st.session_state.month_input
    day = st.session_state.day_input
    inst_no = st.session_state.inst_input

    # Validate if there is an instance for an inst and the given date
    if not validate_inst_no(year, month, day, inst_no, model_dropdown):
        st.write(f"No data available for institution number {inst_no} for the year, month, and day of {year, month, day}")
        return
    
    historical_data = get_historical_data(year, month, day, inst_no, model_dropdown)
    if historical_data is None:
        st.write("No historical data found.")
        return

    st.session_state.pandemic_input = historical_data['pandemic']
    st.session_state.max_temperature_input = historical_data['max_t']
    st.session_state.min_temperature_input = historical_data['min_t']
    st.session_state.precipitation_input = historical_data['precipitation']
    st.session_state.max_Co_input = historical_data['max_Co_lag']
    st.session_state.min_Co_input = historical_data['min_Co_lag']
    st.session_state.mean_Co_input = historical_data['mean_Co_lag']
    st.session_state.max_O3_input = historical_data['max_O3_lag']
    st.session_state.min_O3_input = historical_data['min_O3_lag']
    st.session_state.mean_O3_input = historical_data['mean_O3_lag']
    st.session_state.max_FPM_input = historical_data['max_FPM_lag']
    st.session_state.min_FPM_input = historical_data['min_FPM_lag']
    st.session_state.mean_FPM_input = historical_data['mean_FPM_lag']
    st.session_state.max_No2_input = historical_data['max_No2_lag']
    st.session_state.min_No2_input = historical_data['min_No2_lag']
    st.session_state.mean_No2_input = historical_data['mean_No2_lag']
    st.session_state.max_So2_input = historical_data['max_So2_lag']
    st.session_state.min_So2_input = historical_data['min_So2_lag']
    st.session_state.mean_So2_input = historical_data['mean_So2_lag']


    input_data = [year, month, day, inst_no,
                  historical_data['pandemic'],
                  historical_data['max_t'], historical_data['min_t'], historical_data['precipitation'],
                  historical_data['max_Co_lag'], historical_data['min_Co_lag'], historical_data['mean_Co_lag'],
                  historical_data['max_O3_lag'], historical_data['min_O3_lag'], historical_data['mean_O3_lag'], 
                  historical_data['max_FPM_lag'], historical_data['min_FPM_lag'], historical_data['mean_FPM_lag'], 
                  historical_data['max_No2_lag'], historical_data['min_No2_lag'], historical_data['mean_No2_lag'], 
                  historical_data['max_So2_lag'], historical_data['min_So2_lag'], historical_data['mean_So2_lag']]
    
    columns = X_prediction_3.columns if model_dropdown == 'ED Visits after 3 Days' else X_prediction_7.columns
    # Create the DataFrame
    input_data = pd.DataFrame([input_data], columns=columns)
    
    model = RF_reg_3 if model_dropdown == 'ED Visits after 3 Days' else RF_reg_7    
    # Make prediction using selected model
    predicted_ED_visits_value = model.predict(input_data).round()
    predicted_ED_visits = int(predicted_ED_visits_value[0].round())  


    if year <= latest_year_availableED and month <= latest_month_availableED:
        if model_dropdown == 'ED Visits after 3 Days':
            filtered_df = df_train_3[(df_train_3['year'] == year) & 
                                                       (df_train_3['month'] == month) & 
                                                       (df_train_3['day'] == day) & 
                                                       (df_train_3['inst_no'] == inst_no)]
            
            Actual_ED_visits = filtered_df['smooth_ed_visits']
            Date = filtered_df['date']

            Actual_ED_visit_value = Actual_ED_visits.iloc[0] if not Actual_ED_visits.empty else None
            # Convert Date to datetime and add 3 days
            Date = pd.to_datetime(Date) + pd.Timedelta(days=3)
            Date_value = Date.iloc[0].strftime('%Y-%m-%d') if not Date.empty else None

            seasonal_avg_value = int(get_seasonal_average(month, inst_no, model_dropdown))
            

        else:
            filtered_df = df_train_7[(df_train_7['year'] == year) & 
                                                       (df_train_7['month'] == month) & 
                                                       (df_train_7['day'] == day) & 
                                                       (df_train_7['inst_no'] == inst_no)]
            
            Actual_ED_visits = filtered_df['smooth_ed_visits']
            Date = filtered_df['date']
            Actual_ED_visit_value = Actual_ED_visits.iloc[0] if not Actual_ED_visits.empty else None
            # Convert Date to datetime and add 3 days
            Date = pd.to_datetime(Date) + pd.Timedelta(days=7)
            Date_value = Date.iloc[0].strftime('%Y-%m-%d') if not Date.empty else None

            seasonal_avg_value = int(get_seasonal_average(month, inst_no, model_dropdown))

        if Actual_ED_visit_value is not None:
            st.write(f"**Actual ED Visits:** {int(Actual_ED_visit_value.round())}")
        else:
            st.write("Actual ED Visits: Data unavailable")

        
        if seasonal_avg_value is not None:
            st.write(f"**Seasonal Average ED Visits:** {seasonal_avg_value}")
        else:
            st.write("Seasonal Average ED Visits: No data")
    
        if predicted_ED_visits <= seasonal_avg_value * 1.1:
            color = "green"
        elif predicted_ED_visits <= seasonal_avg_value * 1.25:
            color = "gold"
        else:
            color = "red"
        html_string = f"<b style='font-size: 21px;'>Predicted ED Visits for {Date_value}: <span style='color: {color};'>{predicted_ED_visits}</span></b>"

        st.markdown(html_string, unsafe_allow_html=True)

    else:
    #if year >= latest_year_availableED and month > latest_month_availableED:
        if model_dropdown == 'ED Visits after 3 Days':
            filtered_df = df_lag3[(df_lag3['year'] == year) & 
                              (df_lag3['month'] == month) & 
                              (df_lag3['day'] == day) & 
                              (df_lag3['inst_no'] == inst_no)]
            
            Date = filtered_df['date']
            Date = pd.to_datetime(Date) 
            Date_3 = Date.iloc[0].strftime('%Y-%m-%d') if not Date.empty else None
            st.write(f"Data unvailable for Actual ED Visits for {Date_3}")
            seasonal_avg_value = int(get_seasonal_average(month, inst_no, model_dropdown))
            # Add 3 days
            Date = Date + pd.Timedelta(days=3)
            Date_value = Date.iloc[0].strftime('%Y-%m-%d') if not Date.empty else None


        else:
            filtered_df = df_lag7[(df_lag7['year'] == year) & 
                          (df_lag7['month'] == month) & 
                          (df_lag7['day'] == day) & 
                          (df_lag7['inst_no'] == inst_no)]
            
            Date = filtered_df['date']
            Date = pd.to_datetime(Date) 
            Date_7 = Date.iloc[0].strftime('%Y-%m-%d') if not Date.empty else None
            st.write(f"No Data Available for Actual ED Visits for {Date_7}")
            seasonal_avg_value = int(get_seasonal_average(month, inst_no, model_dropdown))
            # Add 7 days
            Date = Date + pd.Timedelta(days=7)
            Date_value = Date.iloc[0].strftime('%Y-%m-%d') if not Date.empty else None

        
        
        if seasonal_avg_value is not None:
            st.write(f"**Seasonal Average ED Visits:** {seasonal_avg_value}")
        else:
            st.write("Seasonal Average ED Visits: No data")
    
        if predicted_ED_visits <= seasonal_avg_value * 1.1:
            color = "green"
        elif predicted_ED_visits <= seasonal_avg_value * 1.25:
            color = "gold"
        else:
            color = "red"
        html_string = f"<b style='font-size: 21px;'>Predicted ED Visits for {Date_value}: <span style='color: {color};'>{predicted_ED_visits}</span></b>"

        st.markdown(html_string, unsafe_allow_html=True)
        
    st.divider()
    plot_historical_ED_visits(inst_no, model_dropdown)


def plot_historical_ED_visits(inst_no, model_dropdown):
    st.write("#### ED Visits: Trends & Forecasts")
    if model_dropdown == 'ED Visits after 3 Days':
        df_plot = df_predict_3
    else:
        df_plot = df_predict_7
    df_plot = df_plot[df_plot['inst_no'] == inst_no][['smooth_ed_visits', 'Predicted_ED_Visits', 'year', 'month', 'day']]
    df_plot['date'] = pd.to_datetime(df_plot[['year', 'month', 'day']])
    df_plot.drop(columns=['year', 'month', 'day'], inplace=True)

    plt.figure(figsize=(22, 10))
    plt.plot(df_plot['date'], df_plot['smooth_ed_visits'], color='blue', label='Actual')
    plt.plot(df_plot['date'], df_plot['Predicted_ED_Visits'], color='red', label='Prediction')
    plt.xlabel('Date', fontsize=40)
    plt.ylabel('ED Visits', fontsize=40)
    plt.title(f'ED Visits: Trends & Forecasts for Institution {inst_no} - {inst_no_name_map[inst_no]}', fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# Function to create a map with the hospitals location
def hospital_map(inst_input=None, width=300, height=250):
    # Initialize map
    m = folium.Map(location=[43.65335351, -79.40640724], zoom_start=8)
    
    # Add hospital markers
    for i, row in hospital_df.iterrows():
        popup_text = f"{row['inst_no']} - {row['inst_name']}"
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_text,
            tooltip=row["inst_no"],
            icon=folium.Icon(color='blue' if row['inst_no'] != inst_input else 'red')
        ).add_to(m)
    
    # If an institution is selected, zoom in on it
    if inst_input:
        selected_row = hospital_df[hospital_df['inst_no'] == inst_input].iloc[0]
        m.location = [selected_row["latitude"], selected_row["longitude"]]
        m.zoom_start = 100
    
    # Display the map
    folium_static(m, width=width, height=height)

def table(inst_no):
    if model_dropdown == 'ED Visits after 3 Days':
        df_table = df_predict_3[df_predict_3['inst_no'] == inst_no]
        df_table['date'] = pd.to_datetime(df_table[['year', 'month', 'day']])
        df_table = df_table.drop(['year', 'month', 'day', 'Season'], axis=1).reset_index(drop=True)
        df_table = df_table[['date'] + [col for col in df_table.columns if col != 'date']]
        df_table['date'] = pd.to_datetime(df_table['date']).dt.date
        df_table.rename(columns={'smooth_ed_visits': 'Actual_ED_Visits'}, inplace=True)
        df_table['inst_no'] = df_table['inst_no'].astype(int).astype(str)

    else:
        df_table = df_predict_7[df_predict_7['inst_no'] == inst_no]
        df_table['date'] = pd.to_datetime(df_table[['year', 'month', 'day', 'Season']])
        df_table = df_table.drop(['year', 'month', 'day'], axis=1).reset_index(drop=True)
        df_table = df_table[['date'] + [col for col in df_table.columns if col != 'date']]
        df_table['date'] = pd.to_datetime(df_table['date']).dt.date
        df_table.rename(columns={'smooth_ed_visits': 'Actual_ED_Visits'}, inplace=True)
        df_table['inst_no'] = df_table['inst_no'].astype(int).astype(str)
    return df_table
##################################################################################################################################
##################################################################################################################################
#streamlit
#giving a title

# Inject custom CSS to style the title and instructions
st.markdown("""
    <style>
    .title, .instructions {
        background-color: #007BFF;  /* Light blue color */
        color: white;
        padding: 10px;
        }
    .big-font {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the CSS class to the title
st.markdown('<h1 class="title" style="font-size:30px;">Emergency Rooms Visits Tool</h1>', unsafe_allow_html=True)

# Add the instructions with custom styling
st.markdown('''
    <p class="instructions" style="font-size:15px;">
        <strong>Instructions: In the Results tab, select input values from drop-down menu in the boxes. Then, click the 'Predict ED Visits' button for predictions.
    </p>
    ''', unsafe_allow_html=True)



tab1, tab2, tab3= st.tabs(["Note", "Results", "Advanced settings"])

# Define a function to display content with a specific style
def display_section(title, content):
    st.write(f"### {title}")
    st.markdown(f'<p class="big-font">{content}</p>', unsafe_allow_html=True)

with tab1:

    st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True)

    display_section("Predicting Emergency Rooms Visits Using Environmental Factors", "")
    # Display data sources
    # Data Sources
    data_sources = """
    - National Ambulatory Care Reporting System (NACRS)<br>
    - Air pollutants (Ozone, Fine particulate matter, Nitrogen dioxide, Carbon monoxide, and Sulfur dioxide)<br>
    - Multiple yearly weather files
    """
    display_section("Data Sources:", data_sources)

    # ML Models
    display_section("Machine Learning Model for prediction:", "- Random Forest Regression")


    st.markdown(
        """
        <p class="big-font">
        <strong>Independent variables:</strong><br>
        year, month, day, institution number, pandemic,
        maximum and minimum temperature, precipitation,
        maximum, minimum and mean Carbon Monoxide, Ozone, Fine particulate matter, Nitrogen Dioxide, and Sulfur Dioxide.
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <p class="big-font">
        <strong>Dependent variables:</strong><br>
        ED visits Volume
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="big-font">
        <strong>The training data spans from January 2011 to March 2024.</strong><br>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Display how to use the tool
    st.write("### Getting Started with the Tool:")
    st.markdown(
        """
        <p class="big-font">
        <strong>Step1:</strong><br>
        Go to the 'Results' tab and adjust the options
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="big-font">
        <strong>Step2:</strong><br>
        Press the 'Predict ED Visits' button to see the predicted, actual, and seasonal average values of ED visit
        </p>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <p class="big-font">
        <strong>Note 1:</strong><br>
        The seasonal average value of ED volume is calculated using past 2 years available data.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="big-font">
        <strong>Note 2:</strong><br>
            The output results are color-coded accordingly:<br>
            - <span style="color: green;">Green: if the predicted ED visit does not exceed 10% of the seasonal average</span><br>
            - <span style="color: #FFD700;">Yellow: if the predicted ED visit is between 10% and 25% of the seasonal average</span><br>
            - <span style="color: red;">Red: if the predicted ED visit is above 25% of the seasonal average for the selected hospital</span>
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="big-font">
        <strong>Note 3:</strong><br>
        In the 'Advanced Settings' tab, the 'Pandemic' option is specifically for COVID-19, and the sliders are automatically adjusted based on historical values. <br>
        <span> Missing values are set to zero. </span>
        </p>
        """,
        unsafe_allow_html=True
    )

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        # Initialization
        if 'inputs_collected' not in st.session_state:
            st.session_state.inputs_collected = False

        # Collect inputs
        with st.form(key='input_form'):  
            model_dropdown = create_model_dropdown()
            year_input, month_input, day_input, inst_input = create_date_streamlit()

            st.markdown("""
                <style>
                .stButton>button {
                    background-color: #808080;  /* Grey color */
                    color: white;
                    border: none;
                    padding: 0.9em 1em;
                    border-radius: 5px;
                }
                .stButton>button:hover {
                    background-color: #0000CC;  /* Darker grey color */
                </style>
                """, unsafe_allow_html=True)            

            # Button to submit the form
            submit_button = st.form_submit_button(label='Predict ED Visits')

            # Create and display the map
            hospital_map(inst_input)

            # Set the flag when the form is submitted
            if submit_button:
                st.session_state.inputs_collected = True
        
    with col2:
        # Check if inputs are collected and then execute the main logic
        if st.session_state.inputs_collected:
            hospital_no = inst_no_name_map[st.session_state.inst_input]
            st.write(f"### Results for {inst_no_name_map[st.session_state.inst_input]}")
            predict_ED_visits(model_dropdown)
            st.write(f"ED Visits Table for Institution {st.session_state.inst_input} - {inst_no_name_map[st.session_state.inst_input]}")
            df_table = table(st.session_state.inst_input)
            st.dataframe(df_table, height=210)
        else:
            st.write("Select input values and press the 'Predict ED Visits' button to execute.")

with tab3:
    pandemic_input = create_pandemic_streamlit()
    max_temperature_input, min_temperature_input, precipitation_input = create_weather_streamlit()
    max_Co_input, min_Co_input, mean_Co_input, max_O3_input, min_O3_input, mean_O3_input, max_FPM_input, min_FPM_input, mean_FPM_input, max_No2_input, min_No2_input, mean_No2_input, max_So2_input, min_So2_input, mean_So2_input = create_air_quality_streamlit()

