import pandas as pd
import numpy as np


def make_df(in_df, out_df, **columns):
	'''
	Function to create dataframe with engineered features.
	Arguments:
	in_df: base dataframe
	out_df: output dataframe name string
	columns: kwargs: DICTIONARY of column names to apply summary statistics
	
	'''
	# list of columns to not calculate MAD
	no_std_cols= ['DOD', 'Gender', 'Race','RenalDiseaseIndicator',
	   'County', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
	   'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
	   'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
	   'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
	   'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
	   'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
	   'ChronicCond_stroke' ]
	
	# Potential fraud column
	potentialFraud_df = in_df.groupby(['Provider','PotentialFraud'])
	out_df = pd.DataFrame(potentialFraud_df.apply(lambda x: x.name)).reset_index().drop(columns=[0])
	
	# Number unique beneficiaries per provider
	numBene_df = in_df.groupby(['Provider'])['BeneID'].apply(pd.Series.nunique).to_frame(name='numBene').reset_index()
	out_df = out_df.merge(numBene_df, on='Provider')
	
	# Number unique claims per provider
	numClaim_df = Imerge.groupby(['Provider'])['BeneID'].count().to_frame(name='numClaim').reset_index()
	out_df = out_df.merge(numClaim_df, on='Provider')
	
	# Sum reimbursments per provider
	reimb_df = Imerge.groupby(['Provider'])['InscClaimAmtReimbursed'].sum().to_frame(name='InscClaimAmtReimbursed_sum').reset_index()
	out_df = out_df.merge(reimb_df, on='Provider')
	
	# Number unique counties per provider
	Imerge['County'] = Imerge['County'].astype('object')
	numCo_df = Imerge.groupby(['Provider'])[['County']].agg(pd.Series.nunique).reset_index()
	out_df = out_df.merge(numCo_df, on='Provider')
	out_df['County'] = out_df['County'].astype('int64')

	# DOD/Deceased fraction per provider
	Imerge_DOD = Imerge.groupby(['Provider','PotentialFraud','DOD']).size().to_frame(name='size').reset_index()
	Imerge_DOD_pivot = Imerge_DOD.pivot_table(columns='DOD', index=['Provider', 'PotentialFraud'],values='size')
	Imerge_DOD_pivot.reset_index(inplace = True)
	Imerge_DOD_pivot.fillna(0, inplace = True)
	Imerge_DOD_pivot['sum_DOD'] = Imerge_DOD_pivot.iloc[:,-2:].sum(1).values
	out_df['fraction_died'] = (Imerge_DOD_pivot[1].values/Imerge_DOD_pivot['sum_DOD'].values)
	
	# M/F Gender ratio per provider
	Imerge_gender= Imerge.groupby(['Provider','PotentialFraud','Gender']).size().to_frame(name='size').reset_index()
	Imerge_gender_pivot= Imerge_gender.pivot_table(columns='Gender', index=['Provider', 'PotentialFraud'],values='size')
	Imerge_gender_pivot.reset_index(inplace=True)
	out_df['MFgender_ratio']=(Imerge_gender_pivot[1].values/Imerge_gender_pivot[0].values)
	out_df['MFgender_ratio'].fillna(0, inplace = True)
	
	# Race Fraction not white per provider
	Imerge_race = Imerge.groupby(['Provider','PotentialFraud','Race']).size().to_frame(name='size').reset_index()
	Imerge_race_pivot = Imerge_race.pivot_table(columns='Race', index=['Provider', 'PotentialFraud'],values='size')
	Imerge_race_pivot.reset_index(inplace=True)
	Imerge_race_pivot.fillna(0, inplace = True)
	Imerge_race_pivot['Rsum'] = Imerge_race_pivot.iloc[:,-2:].sum(1).values
	out_df['fraction_not_white'] = (Imerge_race_pivot.iloc[:,-3].values/Imerge_race_pivot['Rsum'].values)
	
	# Summary statistics for column names given in kwargs
	for key, value in columns.items():
		if in_df[key].dtypes !='O':
			print(key)
			if key in no_std_cols:
				grp_agg = in_df.groupby(['Provider'])[[value]].agg([(key+'_mean', 'mean')]).reset_index().droplevel(1, axis='columns')
				out_df = out_df.merge(grp_agg, on='Provider')
				out_df.columns.values[-1:] = (key+'_mean') 
			else:
				grp_agg = in_df.groupby(['Provider'])[[value]].agg([(key+'_mean', 'mean'), (key+'_mad', 'mad')]).reset_index().droplevel(1, axis='columns')
				out_df = out_df.merge(grp_agg, on='Provider')
				out_df.columns.values[-2:] = (key+'_mean', key+'_mad')
	
	# Number Unique Admit and Group codes used by providers
	adm_grp_code_cols= ['ClmAdmitDiagnosisCode', 'DiagnosisGroupCode']
	for col in adm_grp_code_cols:
		grp_agg = in_df.groupby(['Provider'])[col].apply(pd.Series.nunique).to_frame(name='_uniq').reset_index()
		out_df = out_df.merge(grp_agg, on='Provider')
		out_df.columns.values[-1:] = col+'_uniq'
		
	# Number Unique Diag codes used by providers
	diagConcat = pd.concat([Imerge['ClmDiagnosisCode_1'],Imerge['ClmDiagnosisCode_2'],Imerge['ClmDiagnosisCode_3'],
				 Imerge['ClmDiagnosisCode_4'],Imerge['ClmDiagnosisCode_5'],
				 Imerge['ClmDiagnosisCode_6'],Imerge['ClmDiagnosisCode_7'],Imerge['ClmDiagnosisCode_8'],
				 Imerge['ClmDiagnosisCode_9'],Imerge['ClmDiagnosisCode_10']], ignore_index=True, sort=False)

	provConcat = pd.concat([Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],
				Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],Imerge['Provider']], ignore_index=True, sort=False)

	diag_provConcat = pd.concat([diagConcat, provConcat], axis=1)

	grp_agg = diag_provConcat.groupby(['Provider'])[0].apply(pd.Series.nunique).to_frame(name='ClmDiagnosisCodes_uniq').reset_index()
	out_df = out_df.merge(grp_agg, on='Provider')
		
	# Number Unique Proc codes used by providers
	# set proc code columns as object
	procConcat = pd.concat([Imerge['ClmProcedureCode_1'].astype("object"),Imerge['ClmProcedureCode_2'].astype("object"),Imerge['ClmProcedureCode_3'].astype("object"),
				 Imerge['ClmProcedureCode_4'].astype("object"),Imerge['ClmProcedureCode_5'].astype("object")], ignore_index=True, sort=False)

	provConcat = pd.concat([Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],Imerge['Provider'],
							Imerge['Provider']], ignore_index=True, sort=False)

	proc_provConcat = pd.concat([procConcat, provConcat], axis=1)

	grp_agg = proc_provConcat.groupby(['Provider'])[0].apply(pd.Series.nunique).to_frame(name='ClmProcedureCodes_uniq').reset_index()
	out_df = out_df.merge(grp_agg, on='Provider')

	# Provider column no longer needed
	out_df.drop(['Provider'], inplace=True, axis=1)
	
	return out_df   
