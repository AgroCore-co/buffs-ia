# =================================================================
# ARQUIVO 2: treinar.py
# OBJETIVO: Ler os CSVs, preparar os dados e treinar o modelo de IA.
# COMO USAR: Execute este arquivo DEPOIS de gerar os dados (python treinar.py).
# =================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from datetime import date

print("\nIniciando o processo de treinamento da IA...")

# --- 1. Carregando os Dados ---
print("Carregando arquivos CSV...")
df_bufalos = pd.read_csv('bufalos.csv', parse_dates=['dt_nascimento'])
df_ciclos = pd.read_csv('ciclos_lactacao.csv', parse_dates=['dt_parto'])
df_ordenhas = pd.read_csv('dados_lactacao.csv')

# --- 2. Preparação dos Dados (Feature Engineering) ---
# Este é o passo mais importante: transformar dados brutos em informações que o modelo entenda.
print("Preparando e juntando os dados...")

# Calcula a produção total por ciclo (nosso alvo de predição)
df_producao_total = df_ordenhas.groupby('id_ciclo_lactacao')['qt_ordenha'].sum().reset_index()
df_producao_total.rename(columns={'qt_ordenha': 'total_leite_ciclo'}, inplace=True)

# Junta a produção com os dados do ciclo para saber quem é a mãe
df_ciclos_prod = pd.merge(df_ciclos, df_producao_total, on='id_ciclo_lactacao')

# Junta com a tabela de búfalos para pegar as informações da MÃE
df_final = pd.merge(df_ciclos_prod, df_bufalos, left_on='id_bufala', right_on='id_bufalo', suffixes=('', '_mae'))

# Agora, a mágica: juntar a tabela de búfalos de novo para pegar os dados dos AVÓS
# Avó Materna (mãe da mãe)
df_final = pd.merge(df_final, df_bufalos[['id_bufalo', 'id_raca', 'potencial_genetico_leite']], 
                    left_on='id_mae', right_on='id_bufalo',
                    suffixes=('', '_avom'), how='left')
df_final.rename(columns={'id_raca': 'id_raca_avom', 'potencial_genetico_leite': 'potencial_avom'}, inplace=True)

# Avô Materno (pai da mãe)
df_final = pd.merge(df_final, df_bufalos[['id_bufalo', 'id_raca', 'potencial_genetico_leite']], 
                    left_on='id_pai', right_on='id_bufalo',
                    suffixes=('', '_avop'), how='left')
df_final.rename(columns={'id_raca': 'id_raca_avop', 'potencial_genetico_leite': 'potencial_avop'}, inplace=True)


# --- 3. Seleção de Features e Alvo ---
# Features (X): As informações que usamos para prever.
# Alvo (y): O que queremos prever.
features = [
    'id_raca',          # Raça da mãe
    'id_raca_avom',     # Raça da avó materna
    'id_raca_avop',     # Raça do avô materno
]
target = 'total_leite_ciclo'

# Lidando com dados faltantes (ex: avós desconhecidos)
df_final[features] = df_final[features].fillna(0) 

X = df_final[features]
y = df_final[target]

print(f"Dataset final criado com {len(df_final)} ciclos de lactação.")

# --- 4. Treinamento do Modelo ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Iniciando treinamento com {len(X_train)} exemplos...")

# RandomForestRegressor é um modelo robusto e ótimo para começar
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# --- 5. Avaliação e Salvamento ---
print("Avaliando o modelo treinado...")
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Treinamento concluído! Erro médio da predição (RMSE): {rmse:.2f} litros")

print("Salvando o modelo em 'modelo_leite.joblib'...")
joblib.dump(model, 'modelo_leite.joblib')

print("\nPronto! O arquivo 'modelo_leite.joblib' foi criado e está pronto para ser usado pela sua API FastAPI.")
