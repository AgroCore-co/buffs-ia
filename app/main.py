# =================================================================
# ARQUIVO 3: main.py (VERSÃO 2 - CONTEXTUAL)
# OBJETIVO: Servir a API que agora entende o contexto de cada propriedade.
# =================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# --- Carregamento do Modelo e Dados Históricos ---
try:
    model = joblib.load('modelo_leite.joblib')
    # Carregamos os dados históricos para calcular as médias de cada propriedade
    df_historico_bufalos = pd.read_csv('bufalos.csv')
    df_historico_ciclos = pd.read_csv('ciclos_lactacao.csv')
    df_historico_ordenhas = pd.read_csv('dados_lactacao.csv')
    
    # Pré-calcula a média de produção por propriedade
    df_prod = df_historico_ordenhas.groupby('id_ciclo_lactacao')['qt_ordenha'].sum().reset_index()
    df_ciclos_prod = pd.merge(df_historico_ciclos, df_prod, on='id_ciclo_lactacao')
    df_completo = pd.merge(df_ciclos_prod, df_historico_bufalos, left_on='id_bufala', right_on='id_bufalo')
    MEDIAS_PROPRIEDADE = df_completo.groupby('id_propriedade')['qt_ordenha'].sum().div(df_completo.groupby('id_propriedade')['id_ciclo_lactacao'].nunique()).to_dict()
    print("Modelo e médias por propriedade carregados.")
except FileNotFoundError:
    model = None
    MEDIAS_PROPRIEDADE = {}
    print("AVISO: Arquivos de modelo ou dados não encontrados. A API não funcionará corretamente.")


# --- DTOs (Data Transfer Objects) ---
class CaracteristicasBufalo(BaseModel):
    id_raca: int

class AcasalamentoInput(BaseModel):
    id_propriedade: int # Agora é obrigatório informar o contexto
    caracteristicas_macho: CaracteristicasBufalo
    caracteristicas_femea: CaracteristicasBufalo
    # Futuramente, podemos adicionar avós aqui
    
# --- Inicialização da API ---
app = FastAPI(title="BUFFS IA API - Contextual", version="1.0.0")

@app.get("/")
def read_root():
    return {"status": "BUFFS IA API Contextual está online"}

@app.post("/prever_potencial_cria")
def prever_potencial_cria(data: AcasalamentoInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de predição não está disponível.")

    # --- LÓGICA DA IA CONTEXTUAL ---
    # Prepara os dados para o modelo (simplificado, usando apenas a média das raças dos pais)
    # Um modelo real usaria dados mais complexos dos avós, etc.
    input_features = {
        "id_propriedade": data.id_propriedade,
        "id_raca": data.caracteristicas_femea.id_raca, # Usando a fêmea como base
        "id_raca_avom": data.caracteristicas_femea.id_raca, # Simulação, idealmente viria no DTO
        "id_raca_avop": data.caracteristicas_macho.id_raca, # Simulação
    }
    df_input = pd.DataFrame([input_features])
    
    # Faz a predição do valor numérico
    previsao_numerica = model.predict(df_input)[0]
    
    # Calcula a classificação contextual
    media_da_propriedade = MEDIAS_PROPRIEDADE.get(data.id_propriedade, previsao_numerica) # Usa a própria previsão se a prop. for nova
    
    classificacao = "Na média da propriedade"
    if previsao_numerica > media_da_propriedade * 1.1: # 10% acima da média
        classificacao = "Potencial Acima da média da propriedade"
    elif previsao_numerica < media_da_propriedade * 0.9: # 10% abaixo da média
        classificacao = "Potencial Abaixo da média da propriedade"

    return {
        "status": "sucesso",
        "predicao_potencial": {
            "producao_leite_litros_estimada": round(previsao_numerica, 2),
            "unidade": "litros/lactação",
            "contexto_propriedade": {
                "id_propriedade": data.id_propriedade,
                "media_producao_local": round(media_da_propriedade, 2),
                "classificacao": classificacao
            },
            "observacao": "Predição contextual baseada no histórico da propriedade."
        }
    }
