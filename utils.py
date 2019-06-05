import mpu
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold, cross_val_score, ShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso, SGDClassifier, HuberRegressor, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, Normalizer, QuantileTransformer, PowerTransformer, PolynomialFeatures, KBinsDiscretizer, StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn import base
from sklearn.ensemble import RandomForestRegressor
import featuretools as ft
from xgboost import XGBRegressor

data_path = Path('../jnscott_tdi_data')

class TextColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_name):
        self.col_name = col_name  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        return X[self.col_name].apply(lambda _string: _string.split(',')).apply(lambda _list: {_string.strip() for _string in _list})
# Modified to return a list of tuples containing strings, for input into MultiLabelBinarizer.

#        return [item for sublist in X[self.col_name].values.tolist() for item in sublist]

def int_parse(x):
    try:
        num = int(re.sub("[^0-9]", "", str(x)))
        return num
    except ValueError:
        return 0

def float_parse(x):
    try:
        num = float(re.sub("[^0-9]", "", str(x)))
        return num
    except ValueError:
        return 0.0

def str_parse(x):
    try:
        id = str(x)
        if id == '':
            return 'NONE'
        else:
            return id
    except ValueError:
        return np.nan

def pickle_all(ndf16, iep16, census_regions, hosps_data, orgs_data, prov_orgs, prov_hosps, prov_sec_specs, hc_star, hc_ans_pct, hc_lin_mean, vocab_dict):
    ndf16.to_pickle('ndf16.df')
    iep16.to_pickle('iep16.df')
    census_regions.to_pickle('census_regions.df')
    hosps_data.to_pickle('hosps_data.df')
    orgs_data.to_pickle('orgs_data.df')
    prov_orgs.to_pickle('prov_orgs.df')
    prov_hosps.to_pickle('prov_hosps.df')
    prov_sec_specs.to_pickle('prov_sec_specs.df')
    hc_star.to_pickle('hc_star.df')
    hc_ans_pct.to_pickle('hc_ans_pct.df')
    hc_lin_mean.to_pickle('hc_lin_mean.df')
    mpu.io.write('vocab_dict.pickle', vocab_dict)
    return print("df's and vocab_dict saved to disk.")

def unpickle_all():
    ndf16 = pd.read_pickle('ndf16.df')
    iep16 = pd.read_pickle('iep16.df')
    census_regions = pd.read_pickle('census_regions.df')
    hosps_data = pd.read_pickle('hosps_data.df')
    orgs_data = pd.read_pickle('orgs_data.df')
    prov_orgs = pd.read_pickle('prov_orgs.df')
    prov_hosps = pd.read_pickle('prov_hosps.df')
    prov_sec_specs = pd.read_pickle('prov_sec_specs.df')
    hc_star = pd.read_pickle('hc_star.df')
    hc_ans_pct = pd.read_pickle('hc_ans_pct.df')
    hc_lin_mean = pd.read_pickle('hc_lin_mean.df')
    vocab_dict = mpu.io.read('vocab_dict.pickle')
    return ndf16, iep16, census_regions, hosps_data, orgs_data, prov_orgs, prov_hosps, prov_sec_specs, hc_star, hc_ans_pct, hc_lin_mean, vocab_dict

def impute_values(ndf16, iep16):
    npi_missing_list = list(set(iep16.npi) - set(ndf16.npi))
    tempndf = ndf16[['npi', 'gndr', 'cred', 'med_sch', 'grd_yr', 'pri_spec', 'num_org_mem', 'assgn', 'pqrs', 'ehr', 'mhi']].drop_duplicates()
    desc = tempndf.describe(include='all')
    gndr_def = desc.loc['top','gndr']
    cred_def = desc.loc['top','cred']
    med_sch_def = desc.loc['top','med_sch']
    grd_yr_def = desc.loc['mean','grd_yr']
    pri_spec_def = desc.loc['top','pri_spec']
    num_org_mem_def = desc.loc['mean','num_org_mem']
    assgn_def = desc.loc['top','assgn']
    pqrs_def = desc.loc['top','pqrs']
    ehr_def = desc.loc['top','ehr']
    mhi_def = desc.loc['top','mhi']
    tempndf = ndf16[['npi', 'gndr', 'cred', 'med_sch', 'grd_yr', 'pri_spec', 'num_org_mem', 'assgn', 'pqrs', 'ehr', 'mhi', 'st']].drop_duplicates()
    desc = tempndf.describe(include='all')
    st_def = desc.loc['top','st']
    var = len(npi_missing_list)
    value_lists = [npi_missing_list, #npi
              [0]*var, #ind_pac_id
              ['None']*var, #ind_enrl_id
              ['None']*var, #lst_nm
              ['None']*var, #frst_nm
              ['None']*var, #mid_nm
              ['None']*var, #suff
              [gndr_def]*var, #gndr
              [cred_def]*var, #cred
              [med_sch_def]*var, #med_sch
              [grd_yr_def]*var, #grd_yr
              [pri_spec_def]*var, #pri_spec
              ['None']*var, #sec_spec_1
              ['None']*var, #sec_spec_2
              ['None']*var, #sec_spec_3
              ['None']*var, #sec_spec_4
              ['None']*var, #sec_spec_all
              ['None']*var, #org_nm
              ['None']*var, #org_pac_id
              [num_org_mem_def]*var, #num_org_mem
              ['None']*var, #adr_ln_1
              ['None']*var, #adr_ln_2
              ['None']*var, #ln_2_sprs
              ['None']*var, #cty
              [st_def]*var, #st
              [0]*var, #zip
              ['None']*var, #phn_numbr
              ['None']*var, #hosp_afl_1
              ['None']*var, #hosp_afl_lbn_1
              ['None']*var, #hosp_afl_2
              ['None']*var, #hosp_afl_lbn_2
              ['None']*var, #hosp_afl_3
              ['None']*var, #hosp_afl_lbn_3
              ['None']*var, #hosp_afl_4
              ['None']*var, #hosp_afl_lbn_4
              ['None']*var, #hosp_afl_5
              ['None']*var, #hosp_afl_lbn_5
              [assgn_def]*var, #assgn
              [pqrs_def]*var, #pqrs
              [ehr_def]*var, #ehr
              [mhi_def]*var] #mhi
    col_names = ndf16.columns
    d = dict(zip(col_names, value_lists))
    imputedf = pd.DataFrame(data=d)
    newndf16 = pd.concat([ndf16, imputedf], ignore_index=True)
    return newndf16

'''
Some providers with scores in the iep16 csv/dataframe do *not* have entries in the ndf16 csv/dataframe, resulting in NaN's in the feature_matrix when the 2 datasets are combined for generating features. 
The ndf16 and iep16 dataframes are loaded and used to determine top values to impute for categorical variables and the mean grd_yr to impute for that variable. All records in 
the ndf16 dataframe have grd_yr values, therefore records are added to ndf16 based on the existence of an npi in iep16 but not in ndf16. A dataframe constructed of the missing npi values and default 
values is finally concatenated with the original dataframe.

'''

def load_data():
    ndf16_converters = {'NPI':int_parse, ' Ind_PAC_ID':int_parse, ' Grd_yr':float_parse, ' org_pac_id':str_parse, ' num_org_mem':int_parse, ' zip':int_parse, ' phn_numbr':int_parse,
                    ' hosp_afl_1':str_parse, ' hosp_afl_2':str_parse, ' hosp_afl_3':str_parse, ' hosp_afl_4':str_parse, ' hosp_afl_5':str_parse, ' sec_spec_1':str_parse, 
                    ' sec_spec_2':str_parse, ' sec_spec_3':str_parse, ' sec_spec_4':str_parse, ' sec_spec_all':str_parse, ' Med_sch':str_parse}
    na_repl_values = {'cred':'UNREPORTED','mhi':'N','ehr':'N','pqrs':'N','med_sch': 'OTHER'}
    iep16_converters = {'NPI':int_parse,
                    ' Ind_PAC_ID':int_parse,
                    ' prf_rate':float_parse,
                    ' patient_count':float_parse}

    print('Step 1/30, import takes ~1  minutes, start time is ', datetime.now())
    ndf16 = pd.read_csv(data_path / 'ndf2016.csv', converters = ndf16_converters).rename(columns=lambda x: x.strip().lower()).fillna(value=na_repl_values)
    iep16 = pd.read_csv(data_path / 'iep2016.csv', converters = iep16_converters).rename(columns=lambda x: x.strip().lower())
    iep16.loc[iep16.invs_msr == 'Y', 'prf_rate'] = 100 - iep16.loc[iep16.invs_msr == 'Y', 'prf_rate']
    ndf16 = impute_values(ndf16, iep16)
    census_regions = pd.read_csv(data_path / 'us_census_bureau_regions_and_divisions.csv').rename(index=str, columns={'State Code':'st'}).rename(columns=lambda x: x.lower())
    hc = pd.read_csv(data_path / 'hcahps_hospital.csv').rename(columns=lambda x: x.strip().lower().replace(' ', '_')).rename(index=str, columns={'provider_id':'hosp_id'})

    ndf16.replace({'Y':'T', 'N':'F'}, inplace=True)
    iep16.replace({'Y':'T', 'N':'F'}, inplace=True)
#    ndf16.replace('N', 'F', inplace=True)
#    iep16.replace('Y', 'T', inplace=True)
#    iep16.replace('N', 'F', inplace=True)

    orgs_data = ndf16[['org_pac_id','num_org_mem', 'st', 'zip']].drop_duplicates()
    hosps_data = hc[['hosp_id', 'state', 'zip_code']].drop_duplicates()

    prov_orgs = ndf16[['npi','org_pac_id']].drop_duplicates()
    prov_hosps = ndf16[['npi', 'hosp_afl_1', 'hosp_afl_2', 'hosp_afl_3', 'hosp_afl_4', 'hosp_afl_5']]
    prov_hosps = prov_hosps.melt(id_vars=['npi'], value_name="hosp_id", value_vars=['hosp_afl_1', 'hosp_afl_2', 'hosp_afl_3', 'hosp_afl_4', 'hosp_afl_5'])\
             [['npi', 'hosp_id']].fillna('NONE').drop_duplicates()
    prov_hosps['hosps_count'] = prov_hosps.npi.map(prov_hosps.groupby('npi')['hosp_id'].count())
    prov_hosps = prov_hosps[(prov_hosps['hosp_id'] == 'NONE') & (prov_hosps['hosps_count'] == 1) | (prov_hosps['hosp_id'] != 'NONE')]\
                 [['npi', 'hosp_id']].sort_values('npi').reset_index(drop=True)
    prov_sec_specs = ndf16[['npi', 'sec_spec_1', 'sec_spec_2', 'sec_spec_3', 'sec_spec_4']]
    prov_sec_specs = prov_sec_specs.melt(id_vars=['npi'], value_name="sec_spec", value_vars=['sec_spec_1', 'sec_spec_2', 'sec_spec_3', 'sec_spec_4'])\
                 [['npi', 'sec_spec']].fillna('NONE').drop_duplicates()
    prov_sec_specs['spec_count'] = prov_sec_specs.npi.map(prov_sec_specs.groupby('npi')['sec_spec'].count())
    prov_sec_specs = prov_sec_specs[(prov_sec_specs['sec_spec'] == 'NONE') & (prov_sec_specs['spec_count'] == 1) | (prov_sec_specs['sec_spec'] != 'NONE')]\
                 [['npi', 'sec_spec']].sort_values('npi').reset_index(drop=True)
    prov_states = ndf16[['npi','st']].drop_duplicates()

    ndf16['states_text'] = ndf16.npi.map(prov_states.iloc[:,:].groupby('npi')['st'].apply(' '.join))
    ndf16['orgs_text'] = ndf16.npi.map(prov_orgs.iloc[:,:].groupby('npi')['org_pac_id'].apply(' '.join))
    ndf16['hosps_text'] = ndf16.npi.map(prov_hosps.iloc[:,:].groupby('npi')['hosp_id'].apply(' '.join))
    ndf16['sec_specs_text'] = ndf16.npi.map(prov_sec_specs.iloc[:,:].groupby('npi')['sec_spec'].apply(' '.join))
    ndf16['chiro_school'] = ndf16['med_sch'].str.contains(".*CHIRO.*")
    ndf16['chiro_school'] = ndf16['chiro_school'].map({True: 1, False: 0})
    ndf16['osteo_school'] = ndf16['med_sch'].str.contains(".*OSTEO.*")
    ndf16['osteo_school'] = ndf16['osteo_school'].map({True: 1, False: 0})
    ndf16['opto_school'] = ndf16['med_sch'].str.contains(".*OPTO.*")
    ndf16['opto_school'] = ndf16['opto_school'].map({True: 1, False: 0})
    ndf16['dent_school'] = ndf16['med_sch'].str.contains(".*DENT.*")
    ndf16['dent_school'] = ndf16['dent_school'].map({True: 1, False: 0})

    cred_vocab = ndf16.cred.unique().tolist()
    sec_spec_vocab = list(set(ndf16.sec_spec_1.unique().tolist()+ndf16.sec_spec_2.unique().tolist()+ndf16.sec_spec_3.unique().tolist()+ndf16.sec_spec_4.unique().tolist()))
    org_pac_id_vocab = ndf16.org_pac_id.unique().tolist()
    pri_spec_vocab = ndf16.pri_spec.unique().tolist()
    hosps_id_vocab = list(set(ndf16.hosp_afl_1.unique().tolist()+ndf16.hosp_afl_2.unique().tolist()+ndf16.hosp_afl_3.unique().tolist()+ndf16.hosp_afl_4.unique().tolist()+ndf16.hosp_afl_5.unique().tolist()))
    measure_id_vocab = iep16.measure_id.unique().tolist()
    st_vocab = ndf16.st.unique().tolist()
    med_sch_vocab = ndf16.med_sch.unique().tolist()

    vocab_dict = {'cred_vocab':cred_vocab,
                  'sec_spec_vocab':sec_spec_vocab,
                  'org_pac_id_vocab':org_pac_id_vocab,
                  'pri_spec_vocab':pri_spec_vocab,
                  'hosps_id_vocab':hosps_id_vocab,
                  'measure_id_vocab':measure_id_vocab,
                  'st_vocab':st_vocab,
                  'med_sch_vocab':med_sch_vocab
                  }

    ndf16 = ndf16[['npi','gndr','assgn','pqrs','ehr','mhi','cred','med_sch','pri_spec','grd_yr','orgs_text','hosps_text','sec_specs_text','states_text','chiro_school','osteo_school','opto_school','dent_school']]
    iep16 = iep16[['npi','live_site_ind','measure_id','measure_title','collection_type','patient_count','prf_rate']]
    hc_star = hc.loc[hc.patient_survey_star_rating != "Not Applicable"][['hosp_id','hcahps_measure_id','patient_survey_star_rating', 'number_of_completed_surveys','survey_response_rate_percent']].replace('Not Available', np.nan).dropna()
    hc_ans_pct = hc.loc[hc.hcahps_answer_percent != "Not Applicable"][['hosp_id','hcahps_measure_id','hcahps_answer_percent', 'number_of_completed_surveys','survey_response_rate_percent']].replace('Not Available', np.nan).dropna()
    hc_lin_mean = hc.loc[hc.hcahps_linear_mean_value != "Not Applicable"][['hosp_id','hcahps_measure_id','hcahps_linear_mean_value', 'number_of_completed_surveys','survey_response_rate_percent']].replace('Not Available', np.nan).dropna()
    ndf16.drop_duplicates(subset='npi', keep="first", inplace=True)
    ndf16.reset_index(drop=True, inplace=True)

    print('Done, Finish time is ', datetime.now())

    return ndf16, iep16, census_regions, hosps_data, orgs_data, prov_orgs, prov_hosps, prov_sec_specs, hc_star, hc_ans_pct, hc_lin_mean, vocab_dict

