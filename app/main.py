# =================================================================
# ARQUIVO: main.py (VERSÃO 1.1 - CAMINHOS CORRIGIDOS)
# OBJETIVO: API para predição individual de produção de leite + 
#           análise de consanguinidade e simulação de acasalamentos.
# =================================================================
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import os # Importa o módulo os para manipulação de caminhos

# Importa módulos locais
from app.models.prediction import (
    fazer_predicao_producao_individual,
    obter_informacoes_femea
)
from app.models.genealogia import (
    criar_arvore_genealogica,
    CalculadorConsanguinidade
)

# --- Bloco de Carregamento ---
model = None
modelo_info = {}
df_historico_bufalos = pd.DataFrame()
df_historico_ciclos = pd.DataFrame()
df_historico_ordenhas = pd.DataFrame()
df_historico_zootecnicos = pd.DataFrame()
df_historico_sanitarios = pd.DataFrame()
df_historico_repro = pd.DataFrame()

try:
    print("Carregando modelo individual e dados para API...")
    
    # <<< CORREÇÃO: Define o caminho para a pasta de dados >>>
    data_path = "dados_benchmark_v1/"

    # Carrega modelo individual
    model = joblib.load('modelo_producao_individual.joblib')
    
    # Carrega informações do modelo
    with open('modelo_producao_individual_info.json', 'r') as f:
        modelo_info = json.load(f)
    
    # <<< CORREÇÃO: Usa o caminho correto para carregar os arquivos >>>
    df_historico_bufalos = pd.read_csv(os.path.join(data_path, 'bufalos.csv'), parse_dates=['dt_nascimento'])
    df_historico_ciclos = pd.read_csv(os.path.join(data_path, 'ciclos_lactacao.csv'), parse_dates=['dt_parto', 'dt_secagem_real'])
    df_historico_ordenhas = pd.read_csv(os.path.join(data_path, 'dados_lactacao.csv'))
    
    try:
        df_historico_zootecnicos = pd.read_csv(os.path.join(data_path, 'dados_zootecnicos.csv'), parse_dates=['dt_registro'])
    except FileNotFoundError:
        df_historico_zootecnicos = pd.DataFrame()
    
    try:
        df_historico_sanitarios = pd.read_csv(os.path.join(data_path, 'dados_sanitarios.csv'), parse_dates=['dt_aplicacao'])
    except FileNotFoundError:
        df_historico_sanitarios = pd.DataFrame()
    
    try:
        df_historico_repro = pd.read_csv(os.path.join(data_path, 'dados_reproducao.csv'), parse_dates=['dt_evento'])
    except FileNotFoundError:
        df_historico_repro = pd.DataFrame()
    
    print("✅ Modelo individual e todos os dados históricos carregados com sucesso!")
    print(f"📊 Features do modelo: {len(modelo_info.get('features', []))}")

except FileNotFoundError as e:
    print(f"❌ ERRO CRÍTICO: Não foi possível carregar um arquivo essencial: {e}.")
    print("A API iniciará com funcionalidade limitada.")

except Exception as e:
    print(f"❌ ERRO INESPERADO DURANTE A INICIALIZAÇÃO: {e}")

# --- Modelos de Dados (DTOs) ---
class PredicaoIndividualInput(BaseModel):
    id_femea: int = Field(..., description="ID único da búfala fêmea.")

class SimulacaoAcasalamentoInput(BaseModel):
    id_macho: int = Field(..., description="ID único do búfalo macho.")
    id_femea: int = Field(..., description="ID único da búfala fêmea.")

class AnaliseGenealogicaInput(BaseModel):
    id_bufalo: int = Field(..., description="ID único do búfalo para análise.")

class PredicaoIndividualResponse(BaseModel):
    id_femea: int
    predicao_litros: float
    classificacao_potencial: str
    percentual_vs_media: float
    producao_media_propriedade: float
    id_propriedade: int
    features_utilizadas: List[str]
    data_predicao: str

class SimulacaoAcasalamentoResponse(BaseModel):
    macho_id: int
    femea_id: int
    consanguinidade_macho: float
    consanguinidade_femea: float
    parentesco_pais: float
    consanguinidade_prole: float
    risco_consanguinidade: str
    recomendacao: str
    predicao_producao_femea: Optional[PredicaoIndividualResponse] = None

class AnaliseGenealogicaResponse(BaseModel):
    id_bufalo: int
    consanguinidade: float
    ancestrais: Dict[str, List[int]]
    descendentes: Dict[str, List[int]]
    risco_genetico: str

class MachosCompativeisResponse(BaseModel):
    femea_id: int
    machos_compatíveis: List[Dict[str, Any]]
    total_encontrados: int
    limite_consanguinidade: float

# --- API Endpoints ---
app = FastAPI(
    title="Buffs IA - Sistema de Predição Individual e Consanguinidade",
    version="1.0.0",
    description="API para predição individual de produção de leite e análise de consanguinidade em búfalos."
)

@app.get("/", tags=["Status"])
def read_root():
    """Endpoint principal que retorna o status da API e do modelo."""
    return {
        "status": "API Operacional - Sistema de Predição Individual + Consanguinidade",
        "versao": "1.0.0",
        "versao_modelo": modelo_info.get("versao", "N/A"),
        "tipo_modelo": modelo_info.get("tipo_modelo", "N/A"),
        "performance_modelo": modelo_info.get("model_performance", {}),
        "features_disponiveis": len(modelo_info.get("features", [])),
        "descricao": modelo_info.get("descricao", "N/A")
    }

@app.post("/predicao-individual", response_model=PredicaoIndividualResponse, tags=["Predição Individual"])
def predicao_producao_individual(data: PredicaoIndividualInput):
    """
    Prevê a produção de leite de uma fêmea em seu próximo ciclo de lactação.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de predição individual não está disponível.")
    
    try:
        resultado = fazer_predicao_producao_individual(
            data.id_femea,
            df_historico_bufalos,
            df_historico_ciclos,
            df_historico_ordenhas,
            df_historico_zootecnicos,
            df_historico_sanitarios,
            df_historico_repro
        )
        
        if resultado is None:
            raise HTTPException(status_code=500, detail="Erro ao processar predição individual.")
        
        return PredicaoIndividualResponse(**resultado)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"ERRO INTERNO: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar a predição.")

@app.post("/simular-acasalamento", response_model=SimulacaoAcasalamentoResponse, tags=["Simulação de Acasalamento"])
def simular_acasalamento(
    data: SimulacaoAcasalamentoInput,
    incluir_predicao_femea: bool = Query(True, description="Se True, inclui predição de produção da fêmea.")
):
    """
    Simula um acasalamento e calcula a consanguinidade da prole.
    """
    try:
        # Cria árvore genealógica
        arvore = criar_arvore_genealogica(df_historico_bufalos)
        calculador = CalculadorConsanguinidade(arvore)
        
        # Simula acasalamento
        simulacao = calculador.simular_acasalamento(data.id_macho, data.id_femea)
        
        # Predição da fêmea (opcional)
        predicao_femea = None
        if incluir_predicao_femea and model is not None:
            predicao_femea = fazer_predicao_producao_individual(
                data.id_femea,
                df_historico_bufalos,
                df_historico_ciclos,
                df_historico_ordenhas,
                df_historico_zootecnicos,
                df_historico_sanitarios,
                df_historico_repro
            )
            if predicao_femea:
                predicao_femea = PredicaoIndividualResponse(**predicao_femea)
        
        # Monta resposta
        response_data = simulacao.copy()
        response_data['predicao_producao_femea'] = predicao_femea
        
        return SimulacaoAcasalamentoResponse(**response_data)
        
    except Exception as e:
        print(f"ERRO NA SIMULAÇÃO: {e}")
        raise HTTPException(status_code=500, detail="Erro ao simular acasalamento.")

@app.post("/analise-genealogica", response_model=AnaliseGenealogicaResponse, tags=["Análise Genealógica"])
def analise_genealogica(data: AnaliseGenealogicaInput):
    """
    Analisa a genealogia de um búfalo e calcula seu coeficiente de consanguinidade.
    """
    try:
        # Cria árvore genealógica
        arvore = criar_arvore_genealogica(df_historico_bufalos)
        calculador = CalculadorConsanguinidade(arvore)
        
        # Calcula consanguinidade
        consanguinidade = calculador.calcular_consanguinidade(data.id_bufalo)
        
        # Obtém ancestrais e descendentes (Esta lógica precisa ser implementada no Calculador)
        ancestrais = {} # Placeholder
        descendentes = {} # Placeholder
        
        # Classifica risco genético
        if consanguinidade > 0.0625:
            risco_genetico = "Alto - Consanguinidade > 6.25%"
        elif consanguinidade > 0.03125:
            risco_genetico = "Médio - Consanguinidade 3.125-6.25%"
        else:
            risco_genetico = "Baixo - Consanguinidade < 3.125%"
        
        return AnaliseGenealogicaResponse(
            id_bufalo=data.id_bufalo,
            consanguinidade=round(consanguinidade * 100, 2),
            ancestrais=ancestrais,
            descendentes=descendentes,
            risco_genetico=risco_genetico
        )
        
    except Exception as e:
        print(f"ERRO NA ANÁLISE GENEALÓGICA: {e}")
        raise HTTPException(status_code=500, detail="Erro ao analisar genealogia.")

@app.get("/machos-compatíveis/{femea_id}", response_model=MachosCompativeisResponse, tags=["Recomendação"])
def encontrar_machos_compatíveis(
    femea_id: int,
    max_consanguinidade: float = Query(6.25, description="Consanguinidade máxima aceitável em %")
):
    """
    Encontra machos compatíveis para uma fêmea baseado na consanguinidade.
    """
    try:
        # Cria árvore genealógica
        arvore = criar_arvore_genealogica(df_historico_bufalos)
        calculador = CalculadorConsanguinidade(arvore)
        
        # Converte percentual para decimal
        max_consanguinidade_decimal = max_consanguinidade / 100
        
        # Encontra machos compatíveis
        machos_compatíveis = calculador.encontrar_machos_compatíveis(
            femea_id, 
            max_consanguinidade_decimal
        )
        
        return MachosCompativeisResponse(
            femea_id=femea_id,
            machos_compatíveis=machos_compatíveis,
            total_encontrados=len(machos_compatíveis),
            limite_consanguinidade=max_consanguinidade
        )
        
    except Exception as e:
        print(f"ERRO NA BUSCA DE MACHOS COMPATÍVEIS: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar machos compatíveis.")

@app.get("/informacoes-femea/{femea_id}", tags=["Informações"])
def obter_informacoes_femea_endpoint(femea_id: int):
    """
    Obtém informações básicas de uma fêmea.
    """
    try:
        info = obter_informacoes_femea(femea_id, df_historico_bufalos)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Fêmea com ID {femea_id} não encontrada.")
        
        return info
        
    except Exception as e:
        print(f"ERRO AO OBTER INFORMAÇÕES DA FÊMEA: {e}")
        raise HTTPException(status_code=500, detail="Erro ao obter informações da fêmea.")

@app.get("/estatisticas-modelo", tags=["Modelo"])
def estatisticas_modelo():
    """
    Retorna estatísticas detalhadas do modelo treinado.
    """
    if not modelo_info:
        raise HTTPException(status_code=503, detail="Informações do modelo não disponíveis.")
    
    return {
        "informacoes_gerais": {
            "versao": modelo_info.get("versao"),
            "tipo_modelo": modelo_info.get("tipo_modelo"),
            "descricao": modelo_info.get("descricao"),
            "mlflow_run_id": modelo_info.get("mlflow_run_id")
        },
        "performance": modelo_info.get("model_performance", {}),
        "features": {
            "total": len(modelo_info.get("features", [])),
            "lista": modelo_info.get("features", [])
        },
        "feature_importance": modelo_info.get("feature_importance", [])
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando Buffs IA - Sistema de Predição Individual + Consanguinidade")
    print("📊 Versão: 1.0.0")
    print("📊 Para visualizar MLflow: mlflow ui")
    uvicorn.run(app, host="0.0.0.0", port=5001)