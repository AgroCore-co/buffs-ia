# =================================================================
# ARQUIVO: treinar.py (VERSÃO 2.2 - CORRIGIDO)
# OBJETIVO: Treinar o modelo de IA usando o ID da propriedade como uma feature.
# =================================================================
import pandas as pd
import numpy as np  # <-- IMPORTANTE: Adicionamos a biblioteca numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

print("\nIniciando o processo de treinamento da IA CONTEXTUAL...")

# --- 1. Carregando os Dados ---
print("Carregando arquivos CSV...")
df_bufalos = pd.read_csv('bufalos.csv')
df_ciclos = pd.read_csv('ciclos_lactacao.csv')
df_ordenhas = pd.read_csv('dados_lactacao.csv')

# --- 2. Preparação dos Dados ---
print("Preparando e juntando os dados...")
df_producao_total = df_ordenhas.groupby('id_ciclo_lactacao')['qt_ordenha'].sum().reset_index()
df_producao_total.rename(columns={'qt_ordenha': 'total_leite_ciclo'}, inplace=True)

df_ciclos_prod = pd.merge(df_ciclos, df_producao_total, on='id_ciclo_lactacao')

# Junta com a tabela de búfalos para pegar as informações da MÃE
df_final = pd.merge(df_ciclos_prod, df_bufalos, left_on='id_bufala', right_on='id_bufalo', suffixes=('', '_mae'))

# Junta com a tabela de búfalos novamente para pegar os dados da AVÓ MATERNA
df_final = pd.merge(df_final, df_bufalos[['id_bufalo', 'id_raca']], left_on='id_mae', right_on='id_bufalo', suffixes=('', '_avom'), how='left')

# Junta com a tabela de búfalos mais uma vez para pegar os dados do AVÔ MATERNO
df_final = pd.merge(df_final, df_bufalos[['id_bufalo', 'id_raca']], left_on='id_pai', right_on='id_bufalo', suffixes=('', '_avop'), how='left')

# --- 3. Seleção de Features e Alvo ---
features = [
    'id_propriedade',
    'id_raca',          # Raça da mãe
    'id_raca_avom',     # Raça da avó materna
    'id_raca_avop',     # Raça do avô materno
]
target = 'total_leite_ciclo'

df_final[features] = df_final[features].fillna(0)
X = df_final[features]
y = df_final[target]

# --- 4. Treinamento do Modelo ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Iniciando treinamento com {len(X_train)} exemplos...")
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# --- 5. Avaliação e Salvamento ---
print("Avaliando o modelo treinado...")
predictions = model.predict(X_test)

# --- CORREÇÃO APLICADA AQUI ---
# Primeiro, calculamos o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, predictions)
# Depois, tiramos a raiz quadrada para obter o RMSE
rmse = np.sqrt(mse)
# -----------------------------

print(f"Treinamento concluído! Erro médio da predição (RMSE): {rmse:.2f} litros")
joblib.dump(model, 'modelo_leite.joblib')
print("Modelo contextual salvo com sucesso como 'modelo_leite.joblib'")
