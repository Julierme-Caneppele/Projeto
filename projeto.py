Análise Exploratória de Dados de Logística
1. Contexto
O código a seguir busca resolver problemas típicos que empresas de logística enfrentam: otimização das rotas de entrega, alocação de entregas nos veículos da frota com capacidade limitada, etc. Os dados são sintetizados de fontes públicas (IBGE, IPEA, etc.) e são representativos dos desafios que a startup enfrenta no dia a dia, especialmente com relação a sua escala.

2. Pacotes e bibliotecas
# importe todas as suas bibliotecas aqui, siga os padrões do PEP8:
!wget -q "https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/main/dataset/deliveries.json" -O deliveries.json 
# - 1º pacotes nativos do python: json, os, etc.;
import json

# - 2º pacotes de terceiros: pandas, seabornm etc.;
import pandas as pd
3. Exploração de dados
# faça o código de exploração de dados:

!wget -q "https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/main/dataset/deliveries.json" -O deliveries.json 
with open('deliveries.json', mode='r', encoding='utf8') as file:
  data = json.load(file)

# - coleta de dados;
example = data[0]
example['name']
example['region']
example['origin']['lat']
example['origin']['lng']
example['vehicle_capacity']
example['deliveries'][0]['point']['lat']
# - wrangling da estrutura;
deliveries_df = pd.DataFrame(data)
hub_origin_df = pd.json_normalize(deliveries_df["origin"]) # normalizar a coluna com uma operação conhecida como flatten
deliveries_df = pd.merge(left=deliveries_df, right=hub_origin_df, how='inner', left_index=True, right_index=True) # juntando ao conjunto principal
deliveries_df = deliveries_df.drop("origin", axis=1) # removendo a coluna origin
deliveries_df = deliveries_df[["name", "region", "lng", "lat", "vehicle_capacity", "deliveries"]]
deliveries_df.rename(columns={"lng": "hub_lng", "lat": "hub_lat"}, inplace=True)
deliveries_exploded_df = deliveries_df[["deliveries"]].explode("deliveries") # normalizar a coluna com explode que transforma cada elemento da lista em uma linha
deliveries_normalized_df = pd.concat([
  pd.DataFrame(deliveries_exploded_df["deliveries"].apply(lambda record: record["size"])).rename(columns={"deliveries": "delivery_size"}),
  pd.DataFrame(deliveries_exploded_df["deliveries"].apply(lambda record: record["point"]["lng"])).rename(columns={"deliveries": "delivery_lng"}),
  pd.DataFrame(deliveries_exploded_df["deliveries"].apply(lambda record: record["point"]["lat"])).rename(columns={"deliveries": "delivery_lat"}),
], axis= 1) # concatenar as colunas
deliveries_df = deliveries_df.drop("deliveries", axis=1) # removendo deliveries
deliveries_df = pd.merge(left=deliveries_df, right=deliveries_normalized_df, how='right', left_index=True, right_index=True) # concatenar dataframes verticalmente
deliveries_df.reset_index(inplace=True, drop=True)
deliveries_df.head()
# - exploração do schema;
deliveries_df.head(n=5) 
deliveries_df.dtypes # Colunas e seus respectivos tipos de dados
deliveries_df.select_dtypes("object").describe().transpose() # Atributos categóricos
deliveries_df.drop(["name", "region"], axis=1).select_dtypes('int64').describe().transpose() # Atributos numéricos
4. Manipulação
# faça o código de manipulação de dados:
hub_df = deliveries_df[["region", "hub_lng", "hub_lat"]]
hub_df = hub_df.drop_duplicates().sort_values(by="region").reset_index(drop=True)
hub_df.head()
import json

import geopy
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="ebac_geocoder") # geocodificação reversa
location = geolocator.reverse("-15.657013854445248, -47.802664728268745")

# - enriquecimento;
from geopy.extra.rate_limiter import RateLimiter # geocodificação nas coordenadas das três regiões(hub) com cidade e bairro

geocoder = RateLimiter(geolocator.reverse, min_delay_seconds=1)

hub_df["coordinates"] = hub_df["hub_lat"].astype(str)  + ", " + hub_df["hub_lng"].astype(str) 
hub_df["geodata"] = hub_df["coordinates"].apply(geocoder)
hub_geodata_df = pd.json_normalize(hub_df["geodata"].apply(lambda data: data.raw))
hub_geodata_df.head()

import numpy as np

hub_geodata_df = hub_geodata_df[["address.town", "address.suburb", "address.city"]]
hub_geodata_df.rename(columns={"address.town": "hub_town", "address.suburb": "hub_suburb", "address.city": "hub_city"}, inplace=True)
hub_geodata_df["hub_city"] = np.where(hub_geodata_df["hub_city"].notna(), hub_geodata_df["hub_city"], hub_geodata_df["hub_town"])
hub_geodata_df["hub_suburb"] = np.where(hub_geodata_df["hub_suburb"].notna(), hub_geodata_df["hub_suburb"], hub_geodata_df["hub_city"])
hub_geodata_df = hub_geodata_df.drop("hub_town", axis=1)
hub_df = pd.merge(left=hub_df, right=hub_geodata_df, left_index=True, right_index=True) # combinando dataframe cidade e bairro ao principal
hub_df = hub_df[["region", "hub_suburb", "hub_city"]]
hub_df.head()
deliveries_df = pd.merge(left=deliveries_df, right=hub_df, how="inner", on="region")
deliveries_df = deliveries_df[["name", "region", "hub_lng", "hub_lat", "hub_city", "hub_suburb", "vehicle_capacity", "delivery_size", "delivery_lng", "delivery_lat"]]
deliveries_df.head()
!wget -q "https://raw.githubusercontent.com/andre-marcos-perez/ebac-course-utils/main/dataset/deliveries-geodata.csv" -O deliveries-geodata.csv #  # geocodificação nas coordenadas das entregas
deliveries_geodata_df = pd.read_csv("deliveries-geodata.csv")
deliveries_geodata_df.head()
deliveries_df = pd.merge(left=deliveries_df, right=deliveries_geodata_df[["delivery_city", "delivery_suburb"]], how="inner", left_index=True, right_index=True)
deliveries_df.head()

# - controle de qualidade;
deliveries_df.info()
deliveries_df.isna().any()
# Geocodificação Reversa
100 * (deliveries_df["delivery_city"].isna().sum() / len(deliveries_df))
100 * (deliveries_df["delivery_suburb"].isna().sum() / len(deliveries_df))
prop_df = deliveries_df[["delivery_city"]].value_counts() / len(deliveries_df)
prop_df.sort_values(ascending=False).head(10)
prop_df = deliveries_df[["delivery_suburb"]].value_counts() / len(deliveries_df)
prop_df.sort_values(ascending=False).head(10)
5. Visualização
# faça o código de visualização de dados:
!pip3 install geopandas;
import geopandas
# download dos dados do mapa do Distrito Federal do site oficial do IBGE
!wget -q "https://geoftp.ibge.gov.br/cartas_e_mapas/bases_cartograficas_continuas/bc100/go_df/versao2016/shapefile/bc100_go_df_shp.zip" -O distrito-federal.zip
!unzip -q distrito-federal.zip -d ./maps
!cp ./maps/LIM_Unidade_Federacao_A.shp ./distrito-federal.shp
!cp ./maps/LIM_Unidade_Federacao_A.shx ./distrito-federal.shx
mapa = geopandas.read_file("distrito-federal.shp")
mapa = mapa.loc[[0]]
mapa.head()

hub_df = deliveries_df[["region", "hub_lng", "hub_lat"]].drop_duplicates().reset_index(drop=True) # mapa dos hubs
geo_hub_df = geopandas.GeoDataFrame(hub_df, geometry=geopandas.points_from_xy(hub_df["hub_lng"], hub_df["hub_lat"]))
geo_hub_df.head()
geo_deliveries_df = geopandas.GeoDataFrame(deliveries_df, geometry=geopandas.points_from_xy(deliveries_df["delivery_lng"], deliveries_df["delivery_lat"])) # mapa das entregas
geo_deliveries_df.head()
# - produza pelo menos duas visualizações;
import matplotlib.pyplot as plt

# cria o plot vazio
fig, ax = plt.subplots(figsize = (50/2.54, 50/2.54))

# plot mapa do distrito federal
mapa.plot(ax=ax, alpha=0.4, color="lightgrey")

# plot das entregas
geo_deliveries_df.query("region == 'df-0'").plot(ax=ax, markersize=1, color="red", label="df-0")
geo_deliveries_df.query("region == 'df-1'").plot(ax=ax, markersize=1, color="blue", label="df-1")
geo_deliveries_df.query("region == 'df-2'").plot(ax=ax, markersize=1, color="seagreen", label="df-2")

# plot dos hubs
geo_hub_df.plot(ax=ax, markersize=30, marker="x", color="black", label="hub")

# plot da legenda
plt.title("Entregas no Distrito Federal por Região", fontdict={"fontsize": 16})
lgnd = plt.legend(prop={"size": 15})
for handle in lgnd.legendHandles:
    handle.set_sizes([50])
    # gráfico de entregas por região
data = pd.DataFrame(deliveries_df[['region', 'vehicle_capacity']].value_counts(normalize=True)).reset_index() # agregação
data.rename(columns={0: "region_percent"}, inplace=True)
data.head()

import seaborn as sns # visualização

with sns.axes_style('whitegrid'):
  grafico = sns.barplot(data=data, x="region", y="region_percent", ci=None, palette="pastel")
  grafico.set(title='Proporção de entregas por região', xlabel='Região', ylabel='Proporção');
  # - adicione um pequeno texto com os insights encontrados;
  A partir dos gráficos percebe-se que o maior volume de entregas concentra-se na regiões df-1 e df-2, com quase 90% do total. O restante fica na regiao df-0 que é menos densa, mas por outro lado é mais extensa territorialmente. Poderíamos supor que a capacidade dos veículos da região df-0 poderia ser reduzida, e agrupada em carros mais leves para agilizar a entrega nas áreas mais afastadas.