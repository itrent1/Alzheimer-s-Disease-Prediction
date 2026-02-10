import pandas as pd

# Загрузка
df_long = pd.read_csv('oasis_longitudinal.csv')
df_cross = pd.read_csv('oasis_cross-sectional.csv')

# 1. Подготовка Longitudinal 
df_long = df_long.sort_values(by=['Subject ID', 'Visit']).groupby('Subject ID').first().reset_index()

# 2. Подготовка Cross-sectional 
df_cross = df_cross.dropna(subset=['CDR']).rename(columns={'Educ': 'EDUC'})

# 3. Единый таргет по шкале CDR
df_long['Diagnosis'] = (df_long['CDR'] > 0).astype(int)
df_cross['Diagnosis'] = (df_cross['CDR'] > 0).astype(int)

# 4. Выбор общих признаков
common_cols = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF', 'Diagnosis']
combined_df = pd.concat([df_long[common_cols], df_cross[common_cols]], axis=0, ignore_index=True)

# 5. Очистка и кодирование
combined_df['SES'] = combined_df['SES'].fillna(combined_df['SES'].median())
combined_df['MMSE'] = combined_df['MMSE'].fillna(combined_df['MMSE'].median())
combined_df['M/F'] = combined_df['M/F'].replace({'M': 1, 'F': 0})

# Сохранение
combined_df.to_csv('combined_alzheimer_data.csv', index=False)
combined_df.drop('Diagnosis', axis=1).to_csv('X_preprocessed.csv', index=False)
combined_df['Diagnosis'].to_csv('y_labels.csv', index=False)

print("Данные успешно объединены!")