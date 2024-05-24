import pickle
import pandas as pd
df = pd.read_csv('geometry_hgcal.txt',sep = ' ')
red_df = df[['u','v','plane','trigLinks']]
red_df.rename(columns={'plane': 'layer'}, inplace=True)

with open('eLink_filts.pkl', 'wb') as f:
    pickle.dump(red_df, f)
