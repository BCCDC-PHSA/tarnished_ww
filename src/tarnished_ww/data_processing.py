## data processing functions
import pandas as pd
from pathlib import Path
from dbfread import DBF
import pyreadr
import statsmodels.api as sm
import numpy as np
import datetime
import math
from datetime import date
from datetime import timedelta
from datetime import datetime
import os

global_wwtp_names_map = {'Annacis Island':'Annacis',
                        'Iona Island':'Iona',
                        'Lions Gate':'Lionsgate',
                        'Lulu Island':'Lulu',
                        'Comox Valley': 'Comox',
                        'Northwest': 'NWL',
                        'Northwest Langley': 'NWL',
                        'Pentincton': 'Penticton'}
global_cases_names_map = {'Fraser Sewerage Area (FSA) without Northwest Langley WWTP / Annacis Island WWTP':'Annacis',
                          'Fraser Sewerage Area (FSA) without Annacis Island / Northwest Langley WWTP':'NWL',
                          'Vancouver Sewerage Area (VSA) / Iona Island WWTP':'Iona',
                          'Lulu Island Sewerage Area (LIWSA) / Lulu Island WWTP':'Lulu',
                          'North Shore Sewerage Area (NSSA) / Lions Gate WWTP':'Lionsgate',
                          'Comox Valley Water Pollution Control Centre':'Comox',
                          'Greater Nanaimo Pollution Control Centre':'Nanaimo',
                          'Kelowna Wastewater Treatment Facility':'Kelowna',
                          'Kamloops Sewage Treatment Centre':'Kamloops',
                          'Penticton Wastewater Treatment Plant':'Penticton',
                          'McLoughlin Point Wastewater Treatment Plant':'Victoria'}

global_infection_names_map = {'Respiratory Syncytial Virus':'rsv',
                              'Metapneumovirus':'metapneuv',
                              'Enterovirus or Rhinovirus':'entv',
                              'Parainfluenza':'paraflu',
                              'Influenza B':'flub',
                              'Influenza A':'flua',
                              'Adenovirus':'adenov',
                              'Bocavirus':'bocav',
                              'Coronavirus Other':'covid_other',
                              'Influenza Untyped':'flu_untyped',
                              'e':'covid', #wwtp target
                              'influ_a':'flua', #wwtp target
                              'influ_b':'flub'}  #wwtp target

global_institution_names_map = {"Richmond General Hospital": "Richmond Hospital",
                                "UBC Health Sciences Centre": "UBC Hospital",
                                "South Similkameen Health Centr":"South Similkameen Health Centre",
                                "Fraser Lake D and T Centre": "Fraser Lake Diagnostic and Treatment Centre",
                                "Lion's Gate Hospital": "Lions Gate Hospital",
                                "Mount St. Joseph Hospital" : "Mount Saint Joseph Hospital",
                                "Nicola Valley Hospital and HC" : "Nicola Valley Hospital and Health Centre",
                                "Cariboo Memorial Hosp and Hlth" : "Cariboo Memorial Hospital",
                                "Ashcroft Hospital and Community" : "Ashcroft Hospital",
                                "Victorian Com. Health Kaslo": "Victorian Community Health Centre of Kaslo",
                                "Lillooet Hospital and Health Cen": "Lillooet Hospital and Health Centre"}

def add_missing_dates_and_fill(group):
        # Generate the full range of dates for the current region/target
        full_range = pd.date_range(
            start=group['date'].min(),
            end=group['date'].max(),
            freq='D'
        )

        group = group.set_index('date').reindex(full_range).reset_index()
        group = group.rename(columns={'index': 'date'})

        # IMPORTANT:
        # Do NOT interpolate wastewater outcomes.
        # Keep cp_ml, load, load_capita as NaN on days without real measurements.
        group['wwtp'] = group['wwtp'].ffill().bfill()
        group['target'] = group['target'].ffill().bfill()

        return group


def merge_datasets_daily(wwtp_df, weather):
        ##merging 
        ww = wwtp_df[['wwtp','date','target','cp_ml','load','load_capita']]
        ww_sorted = ww.sort_values(by=['wwtp', 'date'])
        # Apply the function to each region group
        daily = ww_sorted.groupby(['wwtp','target']).apply(add_missing_dates_and_fill).reset_index(drop=True)
        daily = daily.merge(weather,how='left',on=['wwtp','date'])
        daily = daily.rename(columns={'load':'load_trillion','date':'surveillance_date'})
        return daily

def merge_datasets_weekly(wwtp_df, weather):
    weather_wwtp = wwtp_df.merge(weather,how='left',on=['wwtp','date'])
    weekly = weather_wwtp.groupby(by=['wwtp','target',pd.Grouper(key='date', freq='W')])[['cp_ml',
                                                                                     'load',
                                                                                     'Precip(mm)',
                                                                                     'Temp(C)',
                                                                                     'Specific_Humidity']].mean().reset_index()
    weekly['week_start'] = weekly['date'] - pd.offsets.Week(1) + pd.offsets.Day(1)
    weekly['week_end'] = weekly['date']
    weekly.loc[:,'week_number'] = weekly['week_start'].dt.isocalendar().week
    weekly = weekly.rename(columns={'load':'load_trillion','date': 'surveillance_date'})
    return weekly

def loading_linking_weather_wwtp_data(base_path_weather, base_path_ww):
    #paths
    cwd = os.path.dirname(Path.cwd())
    data_folder = base_path_weather / "Data"
    weather_folder = data_folder / "weather-data"

    #reading weather data
    filepath = weather_folder / "processed_weather_data_distance.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    weather = pd.read_csv(filepath)
    weather['Date'] = pd.to_datetime(weather['Date'])
    weather.rename(columns={'Date':'date','WWTP':'wwtp','DailyPrecip(mm)':'Precip(mm)'},
                   inplace=True) 
    weather.drop(columns = ['Unnamed: 0'], inplace=True)
    weather.replace({'wwtp': global_wwtp_names_map},inplace=True)
    #reading wastewater data
    wwtp_df = pd.read_csv(base_path_ww/ "ww_combined_tetraplex_paws.csv").rename(columns={'collection_date':'date'})
    wwtp_df.replace({'wwtp': global_wwtp_names_map,'target':global_infection_names_map},inplace=True)
    wwtp_df = wwtp_df[(wwtp_df['exclude']==False)] 
    wwtp_df['date'] = pd.to_datetime(wwtp_df['date'])

    merged_daily = merge_datasets_daily(wwtp_df, weather)
    merged_weekly = merge_datasets_weekly(wwtp_df, weather)
    
    merged_daily.to_csv(os.path.join(cwd,"data","interim", "environmental_data_daily.csv"))
    merged_weekly.to_csv(os.path.join(cwd,"data","interim", "environmental_data_weekly.csv"))
    return merged_daily, merged_weekly


def fill_missing_dates_clinical(group):
    # Create a date range from the minimum date to the maximum date for each group
    full_range = pd.date_range(start=group['surveillance_date'].min(), end=group['surveillance_date'].max(), freq='D')
    # Reindex the group with the full range of dates

    group = group.set_index('surveillance_date').reindex(full_range).reset_index()
    group = group.rename(columns={'index': 'surveillance_date'})
    
    # Fill missing 'value' entries with 0
    group['total_cases'] = group['total_cases'].fillna(0)
    group['cases_percentage_child'] =  group['cases_percentage_child'].fillna(0)
    group['cases_percentage_adult'] =  group['cases_percentage_adult'].fillna(0)
    group['cases_percentage_unknown'] =  group['cases_percentage_unknown'].fillna(0)
    group['wwtp'] = group['wwtp'].ffill()
    group['target'] = group['target'].ffill()
    
    # Add the 'group' column back
    #group['group'] = group['group'].iloc[0]
    
    return group

def loading_pre_processing_clinical_date(freq = "D"):
    cwd = os.path.dirname(Path.cwd())
    root_dir = Path('o:\\BCCDC\\Groups\\Analytics\\Projects\\covid_modeling\\06 Projects\\RSV Flu Modelling\\Scripts\\rebeca-python\\wastewater_pred_model')
    filepath = root_dir / "data" / "raw_query"
    
    if freq == "D":
         envpath = os.path.join(cwd,"data","interim", "environmental_data_daily.csv")
    elif freq == "W":
         envpath = os.path.join(cwd,"data","interim", "environmental_data_weekly.csv")
    else:
         Warning("frequency not properly defined")
    
    env_df = pd.read_csv(envpath,index_col=[0])
    env_df['surveillance_date'] = pd.to_datetime(env_df['surveillance_date'])

    files = [file for file in os.listdir(filepath) if (file.endswith(('csv')) and file.startswith('covid'))]
    dfs = []
    for file in files:
        temp = pd.read_csv(os.path.join(filepath, file),index_col=[0]).reset_index()
        dfs.append(temp)
    covid_df = pd.concat(dfs, ignore_index=True)
    covid_df.rename(columns={'wastewater_treatment_plant':'wwtp'},inplace=True)

    covid_df = covid_df[['wwtp','surveillance_date','cases']].replace({'wwtp':global_cases_names_map})
    covid_df['target'] = 'covid'

    covid_df['surveillance_date'] = pd.to_datetime(covid_df['surveillance_date'])
    covid_df = covid_df[['wwtp','surveillance_date',
                         'cases','target']]
    
    files = [file for file in os.listdir(filepath) if (file.endswith(('csv')) and file.startswith('respiratory'))]
    dfs = []
    for file in files:
        temp = pd.read_csv(os.path.join(filepath, file))
        dfs.append(temp)

    cases_df = pd.concat(dfs, ignore_index=True)

    cases_df.rename(columns={'wastewater_treatment_plant':'wwtp',
                             'collection_date':'surveillance_date',
                             'positive':'cases'},inplace=True)
    
    cases_df['surveillance_date']=pd.to_datetime(cases_df['surveillance_date'])
    cases_df['target'] = cases_df['infection_group'].map(global_infection_names_map)
    
    cases_df = cases_df[['wwtp','surveillance_date',
                         'cases','target']].replace({'wwtp':global_cases_names_map})
    
    cases_df = pd.concat([covid_df,cases_df],axis=0,ignore_index=True)
    cases_df['surveillance_date'] = pd.to_datetime(cases_df['surveillance_date'])

    # Apply the function to each group
    cases_df = cases_df.groupby(['wwtp','target']).apply(fill_missing_dates_clinical).reset_index(drop=True)

    return cases_df, env_df

def getting_percentage_age_group(cases_df, label='cases'):
    # Group by region and date, summing over cases
    if 'target' in cases_df.columns:
        total_cases = cases_df.groupby(['wwtp', 'surveillance_date','target'])[label].sum().reset_index().rename(columns={label:f'total_{label}'})
        age_child = cases_df[cases_df['age_group'] == 'child'].rename(columns={label:f'{label}_child'}).reset_index(drop=True).drop(columns = ['age_group'])
        age_adult = cases_df[cases_df['age_group'] == 'adult'].rename(columns={label:f'{label}_adult'}).reset_index(drop=True).drop(columns = ['age_group'])
        age_unknown = cases_df[cases_df['age_group'] == 'unknown'].rename(columns={label:f'{label}_unknown'}).reset_index(drop=True).drop(columns = ['age_group'])

        cases_data = total_cases.merge(age_child, 
                                        on=['wwtp',
                                            'surveillance_date',
                                            'target'], how='left').merge(age_adult, 
                                                                    on =['wwtp',
                                                                        'surveillance_date',
                                                                        'target'], how='left').merge(age_unknown, 
                                                                                                on=['wwtp',
                                                                                                'surveillance_date',
                                                                                                'target'], how='left')
    else:
        total_cases = cases_df.groupby(['wwtp', 'surveillance_date'])[label].sum().reset_index().rename(columns={label:f'total_{label}'})
        age_child = cases_df[cases_df['age_group'] == 'child'].rename(columns={label:f'{label}_child'}).reset_index(drop=True).drop(columns = ['age_group'])
        age_adult = cases_df[cases_df['age_group'] == 'adult'].rename(columns={label:f'{label}_adult'}).reset_index(drop=True).drop(columns = ['age_group'])
        age_unknown = cases_df[cases_df['age_group'] == 'unknown'].rename(columns={label:f'{label}_unknown'}).reset_index(drop=True).drop(columns = ['age_group'])

        cases_data = total_cases.merge(age_child, 
                                        on=['wwtp',
                                            'surveillance_date'], how='left').merge(age_adult, 
                                                                    on =['wwtp',
                                                                        'surveillance_date'], how='left').merge(age_unknown, 
                                                                                                on=['wwtp',
                                                                                                'surveillance_date'], how='left')
    
    cases_data[[f'{label}_child', f'{label}_adult', f'{label}_unknown']] = cases_data[[f'{label}_child', f'{label}_adult', f'{label}_unknown']].fillna(0)

    cases_data[f'{label}_percentage_child'] = cases_data[f'{label}_child']/cases_data[f'total_{label}']
    cases_data[f'{label}_percentage_adult'] = cases_data[f'{label}_adult']/cases_data[f'total_{label}']
    cases_data[f'{label}_percentage_unknown'] = cases_data[f'{label}_unknown']/cases_data[f'total_{label}']

    cases_data = cases_data.drop(columns=[f'{label}_child',f'{label}_adult',f'{label}_unknown'])
    return cases_data

def loading_pre_processing_clinical_date_age(freq = "D", file="covid_test"):
    cwd = os.path.dirname(Path.cwd())
    root_dir = Path('o:\\BCCDC\\Groups\\Analytics\\Projects\\covid_modeling\\06 Projects\\RSV Flu Modelling\\Scripts\\rebeca-python\\wastewater_pred_model')
    filepath = root_dir / "data" / "raw_query"

    if freq == "D":
        envpath = os.path.join(cwd,"data","interim", "environmental_data_daily.csv")
    elif freq == "W":
        envpath = os.path.join(cwd,"data","interim", "environmental_data_weekly.csv")
    else:
        Warning("frequency not properly defined")

    env_df = pd.read_csv(envpath,index_col=[0])
    env_df['surveillance_date'] = pd.to_datetime(env_df['surveillance_date'])
    
    if file.startswith('covid_2'):
        files = [file for file in os.listdir(filepath) if (file.endswith(('csv')) and file.startswith('covid_2'))]
        dfs = []
        for file in files:
            temp = pd.read_csv(os.path.join(filepath, file),index_col=[0]).reset_index()
            dfs.append(temp)
        covid_df = pd.concat(dfs, ignore_index=True)
        covid_df.rename(columns={'wastewater_treatment_plant':'wwtp'},inplace=True)
        covid_df = covid_df.replace({'wwtp':global_cases_names_map})
        covid_df['target'] = 'covid'
        covid_df = covid_df[['wwtp','surveillance_date',
                        'cases','target','age_group']]
        covid_df['total_tests'] = None
    elif file.startswith('covid_t'):
        print("covid_test_data")
        files = [file for file in os.listdir(filepath) if (file.endswith(('csv')) and file.startswith('covid_t'))]
        dfs = []
        for file in files:
            temp = pd.read_csv(os.path.join(filepath, file),index_col=[0]).reset_index()
            dfs.append(temp)
        covid_df = pd.concat(dfs, ignore_index=True)
        covid_df.rename(columns={'wastewater_treatment_plant':'wwtp',
                            'collection_date':'surveillance_date',
                            'positive':'cases'},inplace=True)
        covid_df = covid_df.replace({'wwtp':global_cases_names_map})
        covid_df['target'] = 'covid'
        covid_df = covid_df[['wwtp','surveillance_date',
                        'cases','target']]
        covid_df['total_tests'] = None


    ## respiratory tests files
    files = [file for file in os.listdir(filepath) if (file.endswith(('csv')) and file.startswith('resp'))]
    dfs = []
    for file in files:
        temp = pd.read_csv(os.path.join(filepath, file))
        dfs.append(temp)
    cases_df = pd.concat(dfs, ignore_index=True)
    cases_df.rename(columns={'wastewater_treatment_plant':'wwtp',
                            'collection_date':'surveillance_date',
                            'positive':'cases'},inplace=True)
    cases_df = cases_df.dropna(subset=['wwtp'])
    cases_df['target'] = cases_df['infection_group'].map(global_infection_names_map)
    cases_df = cases_df[['wwtp','surveillance_date',
                        'cases','target','age_group','total_tests']].replace({'wwtp':global_cases_names_map})
    total_tests = cases_df.groupby(['surveillance_date', 'wwtp', 'age_group'], as_index=False)['total_tests'].max()
    total_tests = total_tests.groupby(['surveillance_date', 'wwtp'], as_index=False)['total_tests'].sum().rename(columns={'total_tests': 'total_tests_all_ages'})
    total_tests['surveillance_date'] = pd.to_datetime(total_tests['surveillance_date'])
    cases_df = pd.concat([covid_df,cases_df],axis=0,ignore_index=True)
    cases_df['surveillance_date'] = pd.to_datetime(cases_df['surveillance_date'])
    cases_data = getting_percentage_age_group(cases_df[['wwtp','surveillance_date','cases','target','age_group']])
    cases_data = cases_data[cases_data['target'].isin(['covid','rsv', 'flua'])]    # Apply the function to each group
    cases_data = cases_data.groupby(['wwtp','target']).apply(fill_missing_dates_clinical).reset_index(drop=True)
    cases_data = cases_data.merge(total_tests, on = ['surveillance_date','wwtp'])
    return cases_data, env_df

def linking_env_clinical_data(freq="D"):
    cwd = os.path.dirname(Path.cwd())
    cases_df, env_df = loading_pre_processing_clinical_date_age(freq)
    if freq == "D":
        data = cases_df.merge(env_df, on= ["wwtp","surveillance_date","target"],how = 'right')        
        filename = "final_linked_data_daily.csv"
        filename_pivot = "final_linked_data_2nd_pipeline_daily.csv"
        data[['load_trillion','cp_ml']] = data[['load_trillion', 'cp_ml']].interpolate()#ffill bfill
        data_filter = data[data['target'].isin(['covid','rsv','flua'])]
        temp = data_filter[(data_filter['wwtp']=='Victoria') & (data_filter['surveillance_date'] > pd.Timestamp('2025-06-30'))]
        print("data filter has duplocates?",temp.tail(30))
        data_pivot = data_filter.pivot(index = ['wwtp','surveillance_date',
                                                'total_tests_all_ages'],
                                        columns = 'target', 
                                        values = ['total_cases','load_trillion','cp_ml',
                                        'cases_percentage_child','cases_percentage_adult','cases_percentage_unknown']).reset_index()
    elif freq == "W":
        def weighted_avg_weekly(x):
            total_cases_sum = cases_df.loc[x.index, 'total_cases'].sum()
            if total_cases_sum == 0:
                return 0  # Return 0 if total cases are missing or zero
            return (x * cases_df.loc[x.index,'total_cases']).sum() / total_cases_sum

        # Convert to Weekly Aggregation
        cases_weekly = cases_df.groupby(['wwtp','target', pd.Grouper(key='surveillance_date', freq='W')]).agg({
            'total_cases': 'sum',  # Sum cases per week,
            'total_tests_all_ages': 'sum',  # Sum cases per week
            'cases_percentage_child': lambda x: weighted_avg_weekly(x),
            'cases_percentage_adult': lambda x: weighted_avg_weekly(x),
            'cases_percentage_unknown': lambda x: weighted_avg_weekly(x)
        }).reset_index()

        data = cases_weekly.merge(env_df, on= ["wwtp","surveillance_date","target"],how = 'right')

        filename = "final_linked_data_weekly.csv"
        filename_pivot = "final_linked_data_2nd_pipeline_weekly.csv"
        #data[['load_trillion','cp_ml']] = data[['load_trillion', 'cp_ml']].interpolate()#ffill bfill
        data_filter = data[data['target'].isin(['covid','rsv','flua'])]
        data_pivot = data_filter.pivot(index = ['wwtp','surveillance_date'],
                                        columns = 'target', 
                                        values = ['total_cases','load_trillion','cp_ml',
                                                  'cases_percentage_child','cases_percentage_adult','cases_percentage_unknown']).reset_index()
            
    data_pivot.columns = ["_".join(map(str, col)).strip("_") for col in data_pivot.columns]
    data_pivot.columns = data_pivot.columns.str.replace(" ", "_").str.lower()

    data.to_csv(os.path.join(cwd, "data", "interim", filename))
    data_pivot.to_csv(os.path.join(cwd, "data", "interim", filename_pivot))

    return data_pivot

def replace_st_at_start(text):
    if text.startswith("St "):  # Check if the string starts with "St "
        return "St." + text[2:]  # Replace 'St' with 'St.' and concatenate the rest
    return text  # Return unchanged if "St" isn't at the start

def replace_dict_manually(ed_visits):
    replace_dict= {"&": "and", ' - Emergency': "", " Emergency": "", " Emerg": "",
               r"Hosp\.": "Hospital", "Health Care": "Healthcare", "Reg ": "Regional "}
    for patt, val in replace_dict.items():
        ed_visits['institution_descr'] = ed_visits.institution_descr.str.replace(patt, val, regex = True)
    return ed_visits

def loading_ed_visits_helpers():
    root_dir = Path('o:\\BCCDC\\Groups\\Analytics\\Projects\\covid_modeling\\06 Projects\\RSV Flu Modelling\\Scripts\\rebeca-python\\wastewater_pred_model')
    filepath = root_dir / "data" / "raw_query"
    files = [file for file in os.listdir(filepath) if (file.endswith(('csv')) and file.startswith('age_grouped_ed_visits'))]
    dfs = []
    for file in files:
        temp = pd.read_csv(os.path.join(filepath, file),index_col=[0]).reset_index().dropna()
        dfs.append(temp)
    ed_visits = pd.concat(dfs, ignore_index=True)
    ed_visits['surveillance_date'] = pd.to_datetime(ed_visits['surveillance_date'])
    ed_visits = ed_visits[ed_visits['surveillance_date']>= pd.Timestamp('2020-01-01')]
    hosp_dir = Path('o:\\BCCDC\\Groups\\Analytics\\Projects\\covid_modeling\\06 Projects\\RSV Flu Modelling\\Data\\hosp-data')
    hosp_pc = pd.read_csv(os.path.join(hosp_dir, "hospname_pc_crosswalk_utf8.csv"), index_col=[0])

    dbf_folder = Path("o:\\BCCDC\\Groups\\Analytics\\Projects\\covid_modeling\\06 Projects\\RSV Flu Modelling\\Data\\wwtp_postal_codes")
    dbffile = dbf_folder / "pc_to_wwtp_20230605.dbf"
    table = DBF(dbffile)
    pc_to_wwtp_df = pd.DataFrame(iter(table))

    return ed_visits, hosp_pc, pc_to_wwtp_df

def renaming_hospital_names(ed_visits):
     ed_visits = replace_dict_manually(ed_visits)
     ed_visits['institution_descr']  = ed_visits['institution_descr'].apply(replace_st_at_start)
     ed_visits['institution_descr'] = ed_visits.institution_descr.replace(global_institution_names_map)
     return ed_visits

def mapping_hosp_names_wwtp(hosp_pc, pc_to_wwtp_df):
     wwtp_hosp = pc_to_wwtp_df.merge(hosp_pc.reset_index(), left_on = 'POSTALCODE', right_on = 'PostalCode', how = 'right')
     wwtp_hosp.loc[(wwtp_hosp['Hospital Name'] == 'Victoria General Hospital') & (wwtp_hosp['WWTP'].isna()), 'WWTP'] = \
        'McLoughlin Point Wastewater Treatment Plant'
     return wwtp_hosp

def mapping_edvisits_wwtp(ed_visits, wwtp_hosp):
    ed_wwtp = wwtp_hosp.merge(ed_visits, left_on = 'Hospital Name', right_on = 'institution_descr', how = 'right')
    final_ed_wwtp = ed_wwtp.groupby(['surveillance_date','WWTP','age_group']).sum().reset_index()
    final_ed_wwtp = final_ed_wwtp[['surveillance_date','ed_visits','WWTP', 'age_group']].dropna().replace(global_cases_names_map)\
        .rename(columns={'WWTP':'wwtp','triage_date':'surveillance_date'})
    return final_ed_wwtp

def fill_missing_dates_ed(group):
    # Create a date range from the minimum date to the maximum date for each group
    full_range = pd.date_range(start=group['surveillance_date'].min(), end=group['surveillance_date'].max(), freq='D')
    # Reindex the group with the full range of dates

    group = group.set_index('surveillance_date').reindex(full_range).reset_index()
    group = group.rename(columns={'index': 'surveillance_date'})
    
    # Fill missing 'value' entries with 0
    group['total_ed_visits'] = group['total_ed_visits'].fillna(0)
    group['ed_visits_percentage_child'] =  group['ed_visits_percentage_child'].fillna(0)
    group['ed_visits_percentage_adult'] =  group['ed_visits_percentage_adult'].fillna(0)
    group['wwtp'] = group['wwtp'].ffill()

    # Add the 'group' column back
    #group['group'] = group['group'].iloc[0]
    
    return group

def linking_edvisits_wwtp_cases(ed_wwtp_map, freq = "D"):
    ed_wwtp_map['surveillance_date'] = pd.to_datetime(ed_wwtp_map['surveillance_date'])
    root_dir = Path('o:\\BCCDC\\Groups\\Analytics\\Projects\\covid_modeling\\06 Projects\\RSV Flu Modelling\\Scripts\\rebeca-python\\wastewater_pred_model')
    filepath = root_dir / "data" / "interim"
    if freq == "D":
        env_cases_data = pd.read_csv(os.path.join(filepath, "final_linked_data_2nd_pipeline_daily.csv"), index_col=0)
        filename = "ed_visits_linked_data_daily.csv"
        ed_wwtp_map =  getting_percentage_age_group(ed_wwtp_map, label = 'ed_visits')
        ed_wwtp_map = ed_wwtp_map.groupby(['wwtp']).apply(fill_missing_dates_ed).reset_index(drop=True)
    elif freq == "W":
        env_cases_data = pd.read_csv(os.path.join(filepath, "final_linked_data_2nd_pipeline_weekly.csv"), index_col=0) #change percentage_age_calculation to input main label
        ed_wwtp_map = ed_wwtp_map.groupby(by=['wwtp','age_group',pd.Grouper(key='surveillance_date', freq='W')])[['ed_visits']].sum().reset_index()  
        ed_wwtp_map =  getting_percentage_age_group(ed_wwtp_map, label = 'ed_visits')
        filename = "ed_visits_linked_data_weekly.csv"
    
    env_cases_data['surveillance_date'] = pd.to_datetime(env_cases_data['surveillance_date'])
    ed_wwtp_map['surveillance_date'] = pd.to_datetime(ed_wwtp_map['surveillance_date'])
    data = ed_wwtp_map.merge(env_cases_data, on= ["wwtp","surveillance_date"], how = 'inner')
    filepath = os.path.join(filepath, filename)
    data.to_csv(filepath)

def rename_columns_ed_unlagged(df,features,sufix):
    # final dataframe to model
    rename_dict = {}
    for feature in features:
        rename_dict[feature] = feature+sufix
        
    final_df = df[features+['surveillance_date','wwtp']].rename(columns = rename_dict)
    return final_df

def generate_lag_and_merge(df,features_to_lag,lags_dict):
    """
    Create lagged features for the target variable by region with different lag intervals.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing 'region' and time series data.
        target_column (str): The target column to create lags for.
        lags_dict (dict): A dictionary where keys are lag names and values are the lag intervals (in days or weeks).
        
    Returns:
        pd.DataFrame: The dataframe with lagged features.
    """
    df_copy = df.copy()

    # Apply lagging process by region
    for region in df_copy['wwtp'].unique():
        region_data = df_copy[df_copy['wwtp'] == region].copy()
        for target_column in features_to_lag:
            for lag_name, lag in lags_dict.items():
                df_copy.loc[df_copy['wwtp'] == region, f'{target_column}_{lag_name}'] = region_data[target_column].shift(lag)
    # Drop NaNs by group (region) to avoid data loss
    #df_copy = df_copy.dropna()
    return df_copy

def generating_input_data(cwd, freq="D"):
    if freq == "D":
        output = "all_regions_input_data_ed_daily.csv"
        filepath = os.path.join(cwd, "data", "interim", "ed_visits_linked_data_daily.csv")
        lags_dict = {
                    'lag7': 7,  # 1-week lag
                    #'lag14': 14, # 2-week lag
                    #'lag21': 21, # 3-week lag
                    }
    elif freq == "W":
        output = "all_regions_input_data_ed_weekly.csv"
        filepath = os.path.join(cwd, "data", "interim", "ed_visits_linked_data_weekly.csv")
        lags_dict = {
            'lag7': 1,  # 1-week lag
            #'lag14': 2, # 2-week lag
            #'lag21': 3, # 3-week lag
            }

    data = pd.read_csv(filepath, index_col=0)
    data['surveillance_date'] = pd.to_datetime(data['surveillance_date'])
    columns_to_convert_float = ['total_ed_visits',
                            'ed_visits_percentage_child',
                            'ed_visits_percentage_adult',
                            'ed_visits_percentage_unknown',
                            'total_cases_flua',
                            'total_cases_rsv',
                            'total_cases_covid',
                            'total_tests_all_ages',
                            'cases_percentage_child_flua',
                            'cases_percentage_adult_flua',
                            'cases_percentage_unknown_flua',
                            'cases_percentage_child_rsv',
                            'cases_percentage_adult_rsv',
                            'cases_percentage_unknown_rsv', 
                            'cases_percentage_child_covid',
                            'cases_percentage_adult_covid',
                            'cases_percentage_unknown_covid',
                            'load_trillion_flua',
                            'load_trillion_rsv',
                            'load_trillion_covid']
    data[columns_to_convert_float] = data[columns_to_convert_float].astype(float)
    all_regions_df = generate_lag_and_merge(data,columns_to_convert_float,lags_dict)
    filepath = os.path.join(cwd,"data","processed", output)
    all_regions_df.to_csv(filepath)
    return all_regions_df