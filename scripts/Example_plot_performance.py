import json
from pathlib import Path
import pandas as pd
import numpy as np
import ABRpresto.main
import ABRpresto.utils


pth = 'C:/Data/ABRpresto data/'

df_ABRpresto = ABRpresto.utils.load_fits(pth)

df_Manual = pd.read_csv(pth+'Manual Thresholds.csv')

df_merge = df_Manual.merge(df_ABRpresto, how='left',
                           on=['id', 'timepoint', 'frequency', 'ear']).sort_values(by=['id', 'ear', 'frequency'])
#
# if False:
#     D = pd.read_csv(r"C:\Users\lshaheen\Downloads\Threshold_list_full with alg.csv")
#     DL = pd.read_csv(r"C:\Users\lshaheen\Documents\AOT\Data_list_full_with_meta.csv")
#     DL.drop(index=DL[
#         DL['pth'].str.contains('GJB2-10054_D36_ID6E6C3E1E56_left_ABRtoneConv abr_io 20211001-150744')].index.values,
#             inplace=True)
#     DL.drop(index=DL[
#         DL['pth'].str.contains('GJB2-10146_W18_ID6F20077130_left_ABRtone abr_io 20230207-162343')].index.values,
#             inplace=True)
#     DL.drop(index=DL[
#         DL['pth'].str.contains('PLATFORM-10371_D15_ID6F16095C0E_left_ABRtone abr_io 20230411-135533')].index.values,
#             inplace=True)
#     DL['max_level'] = DL['levels'].str.split('.').str[-2]
#     DL['max_level'] = DL['max_level'].apply(eval)
#     DL['min_level'] = DL['levels'].str.split('.').str[0].str[1:]
#     DL['min_level'] = DL['min_level'].apply(eval)
#     DL['ear'] = DL['pth'].str.split('/').str[-2].str.split('_').str[3]
#
#     D_levs = D.merge(DL[['id', 'timepoint', 'ear', 'max_level', 'min_level']], how='left', on=['id', 'timepoint', 'ear'])
#     D_levs['frequency'] = D_levs['frequency'].astype(int)
#     D_levs.sort_values(by=['id', 'ear', 'frequency'], inplace=True)
#     for v in ['XCsubpublic threshold', 'manual threshold']:
#         D_levs.loc[D_levs[v] == D_levs['min_level'] - 5, v] = -np.inf
#         D_levs.loc[D_levs[v] == D_levs['max_level'] + 5, v] = np.inf
#     Dm.loc[ii, 'threshold'] = Dm.loc[ii, 'max_level'] + 5
#     print(D_levs.head(40).to_string())
#     D_levs.to_csv(r"C:\Users\lshaheen\Documents\AOT\Data_list_full_with_meta_and_levels.csv", index=False)
#     D = D_levs
# else:
#     D = pd.read_csv(r"C:\Users\lshaheen\Documents\AOT\Data_list_full_with_meta_and_levels.csv")
#
# Dm = D.merge(df,how='left', on=['id', 'timepoint', 'frequency', 'ear']).sort_values(by=['id', 'ear', 'frequency'])
# DmSave = Dm[['id', 'timepoint', 'ear', 'frequency', 'min_level', 'max_level', 'manual threshold', 'threshold']].rename(columns = {'threshold': 'ABRpresto threshold'})
# print(DmSave.head(100).to_string())
# DmSave.drop(columns='ABRpresto threshold').to_csv(r"C:\Users\lshaheen\Documents\AOT\Manual Thresholds.csv", index=False)
# DmSave.drop(columns=['manual threshold','min_level', 'max_level']).to_csv(r"C:\Users\lshaheen\Documents\AOT\ABRpresto Thresholds.csv", index=False)
#
# for v in ['threshold', 'XCsubpublic threshold', 'manual threshold']:
#     ii = np.isinf(Dm[v]) & (Dm[v] < 0); Dm.loc[ii, v] = Dm.loc[ii, 'min_level'] - 5
#     ii = np.isinf(Dm[v]) & (Dm[v] > 0); Dm.loc[ii, v] = Dm.loc[ii, 'max_level'] + 5
# Dm['diff'] = Dm['XCsubpublic threshold'] - Dm['threshold']
# Dm['diffMpublic'] = Dm['XCsubpublic threshold'] - Dm['manual threshold']
# Dm['diffMABRpresto'] = Dm['threshold'] - Dm['manual threshold']
# # print(Dm[~Dm['threshold'].isnull()].sort_values('diff').to_string())
# # print(Dm[~Dm['threshold'].isnull() & (np.abs(Dm['diff'])>.1)].sort_values('diff').to_string())
# # print(Dm[~Dm['threshold'].isnull() & (np.abs(Dm['diff'])>.1) & ~np.isinf(Dm['diff'])].sort_values('diff').to_string())
#
# Dmm=Dm[~Dm['threshold'].isnull() & ~np.isinf(Dm['diff'])]
# (np.abs(Dmm['diffMABRpresto']) <= 10).sum()/len(D)
# (np.abs(Dmm['diffMpublic']) <= 10).sum()/len(D)

