import pandas as pd

TRAIN_DATASET = "../samples/contracts_sample.csv"

df = pd.read_csv(TRAIN_DATASET, sep=';')

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

df['Preço contratual'] = df['Preço contratual'].str.replace(' €', '').str.replace(',', '').astype(float)
df['Preço total efetivo'] = df['Preço total efetivo'].str.replace(' €', '').str.replace(',', '').astype(float)
df['Prazo de execução'] = df['Prazo de execução'].str.replace(' dias', '').str.replace(',', '').astype(int)

df = df.fillna(0)

df['Data de publicação'] = pd.to_datetime(df['Data de publicação'], errors='coerce', dayfirst=True)
df['Data de celebração'] = pd.to_datetime(df['Data de celebração'], errors='coerce', dayfirst=True)
df['Data de fecho'] = pd.to_datetime(df['Data de fecho'], errors='coerce', dayfirst=True)

df['Incrementos superiores a 15%'] = df['Incrementos superiores a 15%'].astype(int)

df

# df.to_csv("filtered_dataset.csv",index=False)