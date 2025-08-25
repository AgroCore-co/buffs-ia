# =================================================================
# ARQUIVO: main.py (VERSÃO 3 - ALINHADO COM NESTJS)
# OBJETIVO: Receber IDs de búfalos, buscar suas características e prever o potencial da cria.
# =================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# --- Carregamento do Modelo e Dados Históricos ---
# Esta seção funciona como um "banco de dados em memória" para a IA.
try:
    model = joblib.load('modelo_leite.joblib')
    
    # Carregamos os dados históricos para buscar informações e calcular médias.
    df_historico_bufalos = pd.read_csv('bufalos.csv')
    df_historico_ciclos = pd.read_csv('ciclos_lactacao.csv')
    df_historico_ordenhas = pd.read_csv('dados_lactacao.csv')
    
    # Pré-calcula a média de produção por propriedade para o contexto.
    df_prod = df_historico_ordenhas.groupby('id_ciclo_lactacao')['qt_ordenha'].sum().reset_index()
    df_ciclos_prod = pd.merge(df_historico_ciclos, df_prod, on='id_ciclo_lactacao')
    df_completo = pd.merge(df_ciclos_prod, df_historico_bufalos, left_on='id_bufala', right_on='id_bufalo')
    MEDIAS_PROPRIEDADE = df_completo.groupby('id_propriedade')['qt_ordenha'].sum().div(df_completo.groupby('id_propriedade')['id_ciclo_lactacao'].nunique()).to_dict()
    
    print("Modelo e dados históricos carregados com sucesso.")
except FileNotFoundError:
    model = None
    MEDIAS_PROPRIEDADE = {}
    df_historico_bufalos = pd.DataFrame() # Garante que o dataframe exista mesmo se falhar
    print("AVISO: Arquivos de modelo ou dados não encontrados. A API não funcionará corretamente.")


# --- DTOs (Data Transfer Objects) ---
# AGORA ESPERA OS IDs, ASSIM COMO O DTO DO NESTJS
class AcasalamentoInput(BaseModel):
    id_macho: int
    id_femea: int
    
# --- Inicialização da API ---
app = FastAPI(title="BUFFS IA API", version="1.1.0")

@app.get("/")
def read_root():
    return {"status": "BUFFS IA API Final está online"}

@app.post("/prever_potencial_cria")
def prever_potencial_cria(data: AcasalamentoInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de predição não está disponível.")

    # --- LÓGICA DE BUSCA E PREDIÇÃO ---
    try:
        # 1. Busca os dados da fêmea e do macho usando os IDs recebidos.
        femea = df_historico_bufalos.loc[df_historico_bufalos['id_bufalo'] == data.id_femea].iloc[0]
        macho = df_historico_bufalos.loc[df_historico_bufalos['id_bufalo'] == data.id_macho].iloc[0]

        # Busca os dados dos avós da futura cria (pais da fêmea)
        avo_paterno = df_historico_bufalos.loc[df_historico_bufalos['id_bufalo'] == femea['id_pai']].iloc[0] if pd.notna(femea['id_pai']) else None
        avo_materna = df_historico_bufalos.loc[df_historico_bufalos['id_bufalo'] == femea['id_mae']].iloc[0] if pd.notna(femea['id_mae']) else None

    except IndexError:
        # Se .iloc[0] falhar, significa que o ID não foi encontrado.
        raise HTTPException(status_code=404, detail="ID do macho ou da fêmea não encontrado nos dados históricos.")

    # 2. Monta o DataFrame de input para o modelo com as CARACTERÍSTICAS encontradas.
    input_features = {
        "id_propriedade": femea['id_propriedade'], # O contexto vem da propriedade da fêmea
        "id_raca": femea['id_raca'],
        "id_raca_avom": avo_materna['id_raca'] if avo_materna is not None else 0,
        "id_raca_avop": avo_paterno['id_raca'] if avo_paterno is not None else 0,
    }
    df_input = pd.DataFrame([input_features])
    
    # 3. Faz a predição do valor numérico.
    previsao_numerica = model.predict(df_input)[0]
    
    # 4. Calcula a classificação contextual.
    media_da_propriedade = MEDIAS_PROPRIEDADE.get(femea['id_propriedade'], previsao_numerica)
    
    classificacao = "Na média da propriedade"
    if previsao_numerica > media_da_propriedade * 1.1:
        classificacao = "Potencial Acima da média da propriedade"
    elif previsao_numerica < media_da_propriedade * 0.9:
        classificacao = "Potencial Abaixo da média da propriedade"

    return {
        "status": "sucesso",
        "predicao_potencial": {
            "producao_leite_litros_estimada": round(previsao_numerica, 2),
            "unidade": "litros/lactação",
            "contexto_propriedade": {
                "id_propriedade": int(femea['id_propriedade']),
                "media_producao_local": round(media_da_propriedade, 2),
                "classificacao": classificacao
            },
            "observacao": "Predição contextual baseada no histórico da propriedade."
        }
    }
