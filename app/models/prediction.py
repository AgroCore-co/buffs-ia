import joblib
import pandas as pd
from datetime import date

# Carrega o modelo uma vez quando este módulo é importado
try:
    model = joblib.load('modelo_leite.joblib')
except FileNotFoundError:
    model = None

def fazer_predicao_leite(caracteristicas_femea: dict):
    if model is None:
        return None # Ou lançar um erro

    df_input = pd.DataFrame([caracteristicas_femea])
    df_input['mae_idade_dias'] = (pd.to_datetime('today') - pd.to_datetime(df_input['mae_dt_nascimento'])).dt.days
    df_input.drop('mae_dt_nascimento', axis=1, inplace=True)
    
    colunas_modelo = ['mae_id_raca', 'avom_id_raca', 'avop_id_raca', 'mae_idade_dias']
    df_input = df_input[colunas_modelo]

    previsao = model.predict(df_input)[0]
    return previsao