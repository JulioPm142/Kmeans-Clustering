import pandas as pd

# Carregar o arquivo CSV
file_path = 'Spotify_Dataset_V3.csv'
df = pd.read_csv(file_path, delimiter=';')

# Remover linhas duplicadas com base na coluna 'id'
df = df.drop_duplicates(subset='id')

# Filtrar as músicas com rank igual ou inferior a 200
df = df[df['Rank'] <= 200]

# Selecionar apenas os primeiros 200 registros
df = df.head(200)

# Remover as colunas especificadas
columns_to_drop = ['# of Artist', 'Artist (Ind.)', '# of Nationality', 'Nationality', 'Continent', 'Points (Total)', 'Points (Ind for each Artist/Nat)', 'Song URL']
df = df.drop(columns=columns_to_drop)

# Salvar o arquivo CSV resultante
output_file_path = 'Spotify_Dataset_Limpo.csv'
df.to_csv(output_file_path, index=False, sep=';')

print(f'Arquivo salvo como {output_file_path}')
