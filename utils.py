import pandas as pd
import numpy as np
from configs import TRAIN_DATASET
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def load_dataset(file_path):
	df = pd.read_csv(file_path, sep=';')

	df = df.drop(columns=[
		'Estado', 'Descrição', 'Tipo de contrato', 'Tipo de procedimento', 'Anúncio (ID)', 'Fundamentação', 
		'Fundamentação da necessidade de recurso ao ajuste direto','Procedimento centralizado', 'Causas das alterações do prazo', 
		'Causas das alterações do preço', 'Documentos (IDs)', 'Observações', 'Local de execução (País, Distrito, Concelho)', 
		'Entidade adjudicante', 'Entidade adjudicatária', 'Convidados (ID|NIF|Descrição)', 'Concorrentes (ID|NIF|Descrição)', 
		'CPV', 'Objeto do contrato', 'Entidade adjudicante (ID)'
		])

	df.head()

	df['Data de publicação'] = pd.to_datetime(df['Data de publicação'], errors='coerce', dayfirst=True)
	df['Data de celebração'] = pd.to_datetime(df['Data de celebração'], errors='coerce', dayfirst=True)
	df['Data de fecho'] = pd.to_datetime(df['Data de fecho'], errors='coerce', dayfirst=True)

	df['Preço contratual'] = df['Preço contratual'].str.replace(' €', '').str.replace('.', '').str.replace(',', '.').astype(float)
	df['Preço total efetivo'] = df['Preço total efetivo'].str.replace(' €', '').str.replace('.', '').str.replace(',', '.').astype(float)
	df['Prazo de execução'] = df['Prazo de execução'].str.replace(' dias', '').str.replace('.', '').str.replace(',', '.').astype(float)

	df = df.fillna(0)

	return df


def load_dataset2(file_path):
	df = pd.read_csv(file_path, sep=';')

	df['Preço contratual'] = df['Preço contratual'].str.replace(' €', '').str.replace('.', '').str.replace(',', '.').astype(float)
	df['Preço total efetivo'] = df['Preço total efetivo'].str.replace(' €', '').str.replace('.', '').str.replace(',', '.').astype(float)
	df['Prazo de execução'] = df['Prazo de execução'].str.replace(' dias', '').str.replace('.', '').str.replace(',', '.').astype(float)

	df = df.select_dtypes(include=np.number)

	# normalized_data = (df - df.min()) / (df.max() - df.min())

	# normalized_data = normalized_data.fillna(0)

	# Padronização
	scaler = StandardScaler()
	standardized_data = scaler.fit_transform(df)
	standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
	standardized_df = standardized_df.fillna(0)


	# Normalização
	normalizer = MinMaxScaler()
	normalized_data = normalizer.fit_transform(df)
	normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
	normalized_df = normalized_df.fillna(0)

	normalized_df.fillna(0)

	# print("Padronização:\n", standardized_df)
	# print("Normalização:\n", normalized_df.to_numpy(dtype=np.number))

	return [standardized_df, normalized_df]


# [standardized_df, normalized_df] = load_dataset2(TRAIN_DATASET)

# print(normalized_df)