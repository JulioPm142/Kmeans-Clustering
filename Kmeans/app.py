import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # Usar o backend 'Agg' para evitar o uso do Tkinter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importação adicional para gráficos 3D

# Carregar o arquivo CSV
file_path = 'Spotify_Dataset_Limpo.csv'
df = pd.read_csv(file_path, delimiter=';')

# Selecionar as colunas para o clustering
features = ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence']
X = df[features]

# Normalizar os dados para melhorar a performance do K-means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar o algoritmo K-means
n_clusters = 7  # Ajuste o número de clusters conforme necessário
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Adicionar os rótulos dos clusters ao DataFrame original
df['Cluster'] = y_kmeans

# Reduzir as dimensões para 3D usando PCA para visualização
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Salvar o arquivo CSV resultante com os clusters
output_file_path = 'DataSetLimpo_com_Clusters.csv'
df.to_csv(output_file_path, index=False, sep=';')

print(f'Arquivo salvo como {output_file_path}')

# Estatísticas do K-means
centroids = kmeans.cluster_centers_
cluster_sizes = pd.Series(y_kmeans).value_counts()
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(X_scaled, y_kmeans)

# Definir cores personalizadas para cada cluster
cluster_colors = ['#E0DC00', '#E10601', '#0EE101', '#000CEA', '#00E1D3', '#F56C00', '#B900EA']

# Salvar a imagem dos clusters em 3D com cores personalizadas para cada cluster
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotar os clusters
for cluster_num, color in enumerate(cluster_colors):
    indices = y_kmeans == cluster_num
    ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], label=f'Cluster {cluster_num} (Tamanho: {cluster_sizes[cluster_num]})', color=color)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('Clusters de músicas (K-means com PCA em 3D)')

# Adicionar a legenda
ax.legend(title="Clusters")

# Adicionar as estatísticas ao gráfico
textstr = '\n'.join((
    f'Inércia (WCSS): {inertia:.2f}',
    f'Métrica de Silhueta Média: {silhouette_avg:.2f}'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
          verticalalignment='top', bbox=props)

image_path = 'clusters_pca_3d_com_estatisticas.png'
plt.savefig(image_path)  # Salvar a imagem
plt.close()

print(f'Imagem salva como {image_path}')
