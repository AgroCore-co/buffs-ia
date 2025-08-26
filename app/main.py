# =================================================================
# ARQUIVO: main.py (VERSÃO 1.1.0 - INTEGRAÇÃO SUPABASE)
# OBJETIVO: API para predição individual de produção de leite + 
#           análise de consanguinidade e simulação de acasalamentos.
#           NOVA FUNCIONALIDADE: Conecta diretamente ao Supabase
# =================================================================
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Importa módulos locais
from app.models.prediction import (
    fazer_predicao_producao_individual,
    obter_informacoes_femea
)
from app.models.genealogia import (
    criar_arvore_genealogica,
    CalculadorConsanguinidade
)
from app.database import supabase_db

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Bloco de Carregamento ---
model = None
modelo_info = {}
df_historico_bufalos = pd.DataFrame()
df_historico_ciclos = pd.DataFrame()
df_historico_ordenhas = pd.DataFrame()
df_historico_zootecnicos = pd.DataFrame()
df_historico_sanitarios = pd.DataFrame()
df_historico_repro = pd.DataFrame()

# Flags de controle
usando_supabase = False
dados_csv_disponiveis = False

try:
    print("🚀 Iniciando Buffs IA v1.1.0 - Integração Supabase")
    print("="*60)
    
    # 1. Carrega modelo individual
    try:
        model = joblib.load('modelo_producao_individual.joblib')
        with open('modelo_producao_individual_info.json', 'r') as f:
            modelo_info = json.load(f)
        print("✅ Modelo individual carregado com sucesso")
    except FileNotFoundError:
        print("⚠️ Modelo não encontrado - funcionalidade de predição limitada")
    
    # 2. Testa conexão com Supabase
    print("\n🔌 Testando conexão com Supabase...")
    if supabase_db.test_connection():
        print("✅ Conexão com Supabase ativa - carregando dados reais")
        usando_supabase = True
        
        try:
            df_historico_bufalos = supabase_db.get_bufalos_data()
            df_historico_ciclos = supabase_db.get_ciclos_lactacao()
            df_historico_ordenhas = supabase_db.get_dados_lactacao()
            df_historico_zootecnicos = supabase_db.get_dados_zootecnicos()
            df_historico_sanitarios = supabase_db.get_dados_sanitarios()
            df_historico_repro = supabase_db.get_dados_reproducao()
            
            print(f"📊 Carregados do Supabase:")
            print(f"   • {len(df_historico_bufalos)} búfalos")
            print(f"   • {len(df_historico_ciclos)} ciclos de lactação")
            print(f"   • {len(df_historico_ordenhas)} registros de ordenha")
            print(f"   • {len(df_historico_zootecnicos)} registros zootécnicos")
            print(f"   • {len(df_historico_sanitarios)} registros sanitários")
            print(f"   • {len(df_historico_repro)} registros reprodutivos")
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar alguns dados do Supabase: {e}")
    
    else:
        print("⚠️ Supabase indisponível - tentando dados CSV...")
        usando_supabase = False
    
    # 3. Fallback para dados CSV (se Supabase indisponível ou como complemento)
    if not usando_supabase or len(df_historico_bufalos) == 0:
        try:
            print("\n📁 Carregando dados dos arquivos CSV...")
            df_csv_bufalos = pd.read_csv('bufalos.csv', parse_dates=['dt_nascimento'])
            df_csv_ciclos = pd.read_csv('ciclos_lactacao.csv', parse_dates=['dt_parto', 'dt_secagem_real'])
            df_csv_ordenhas = pd.read_csv('dados_lactacao.csv')
            
            try:
                df_csv_zootecnicos = pd.read_csv('dados_zootecnicos.csv', parse_dates=['dt_registro'])
            except FileNotFoundError:
                df_csv_zootecnicos = pd.DataFrame()
            
            try:
                df_csv_sanitarios = pd.read_csv('dados_sanitarios.csv', parse_dates=['dt_aplicacao'])
            except FileNotFoundError:
                df_csv_sanitarios = pd.DataFrame()
            
            try:
                df_csv_repro = pd.read_csv('dados_reproducao.csv', parse_dates=['dt_evento'])
            except FileNotFoundError:
                df_csv_repro = pd.DataFrame()
            
            # Se Supabase não funcionou, usa CSV como principal
            if not usando_supabase:
                df_historico_bufalos = df_csv_bufalos
                df_historico_ciclos = df_csv_ciclos
                df_historico_ordenhas = df_csv_ordenhas
                df_historico_zootecnicos = df_csv_zootecnicos
                df_historico_sanitarios = df_csv_sanitarios
                df_historico_repro = df_csv_repro
                
                print(f"📊 Carregados dos CSVs:")
                print(f"   • {len(df_historico_bufalos)} búfalos")
                print(f"   • {len(df_historico_ciclos)} ciclos de lactação")
                print(f"   • {len(df_historico_ordenhas)} registros de ordenha")
            
            dados_csv_disponiveis = True
            print("✅ Dados CSV mantidos para treinamento futuro da IA")
            
        except FileNotFoundError as e:
            print(f"❌ ERRO: Nem Supabase nem CSV estão disponíveis: {e}")
            print("💡 Execute: python gerar_dados.py para criar dados de teste")
    
    # 4. Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DA INICIALIZAÇÃO:")
    print("="*60)
    print(f"🔌 Supabase: {'✅ Ativo' if usando_supabase else '❌ Indisponível'}")
    print(f"📁 Dados CSV: {'✅ Disponíveis' if dados_csv_disponiveis else '❌ Indisponíveis'}")
    print(f"🧠 Modelo IA: {'✅ Carregado' if model else '❌ Não carregado'}")
    print(f"📈 Performance: {modelo_info.get('model_performance', {}).get('r2', 'N/A')}")
    print(f"🐃 Total búfalos: {len(df_historico_bufalos)}")
    
    fonte_dados = "Supabase (dados reais)" if usando_supabase else "CSV (dados sintéticos)"
    print(f"📊 Fonte de dados ativa: {fonte_dados}")
    print("="*60)

except Exception as e:
    print(f"❌ ERRO CRÍTICO DURANTE A INICIALIZAÇÃO: {e}")
    import traceback
    traceback.print_exc()

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

# --- API FastAPI ---
app = FastAPI(
    title="Buffs IA - Sistema de Predição Individual e Consanguinidade",
    version="1.1.0",
    description="API para predição individual de produção de leite e análise de consanguinidade em búfalos. Integração com Supabase."
)

# Configurar CORS para comunicação com NestJS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/", tags=["Status"])
def read_root():
    """Endpoint principal que retorna o status da API e do modelo."""
    return {
        "status": "API Operacional - Sistema de Predição Individual + Consanguinidade",
        "versao": "1.1.0 - Integração Supabase",
        "supabase_ativo": usando_supabase,
        "dados_csv_disponiveis": dados_csv_disponiveis,
        "fonte_dados": "Supabase (dados reais)" if usando_supabase else "CSV (dados sintéticos)",
        "versao_modelo": modelo_info.get("versao", "N/A"),
        "tipo_modelo": modelo_info.get("tipo_modelo", "N/A"),
        "performance_modelo": modelo_info.get("model_performance", {}),
        "features_disponiveis": len(modelo_info.get("features", [])),
        "total_bufalos": len(df_historico_bufalos),
        "descricao": modelo_info.get("descricao", "N/A")
    }

@app.get("/status/conexoes", tags=["Status"])
def status_conexoes():
    """Verifica status das conexões."""
    return {
        "supabase": {
            "ativo": usando_supabase,
            "teste_conexao": supabase_db.test_connection() if supabase_db else False
        },
        "dados_csv": {
            "disponiveis": dados_csv_disponiveis,
            "arquivos_encontrados": {
                "bufalos.csv": len(df_historico_bufalos) > 0,
                "ciclos_lactacao.csv": len(df_historico_ciclos) > 0,
                "dados_lactacao.csv": len(df_historico_ordenhas) > 0
            }
        },
        "modelo_ia": {
            "carregado": model is not None,
            "performance": modelo_info.get("model_performance", {})
        }
    }

@app.post("/predicao-individual", response_model=PredicaoIndividualResponse, tags=["Predição Individual"])
def predicao_producao_individual(data: PredicaoIndividualInput):
    """Prevê a produção de leite de uma fêmea em seu próximo ciclo de lactação."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo de predição individual não está disponível.")
    
    try:
        # Se usando Supabase, verifica se o búfalo existe
        if usando_supabase:
            bufalo_info = supabase_db.get_bufalo_by_id(data.id_femea)
            if not bufalo_info:
                raise HTTPException(status_code=404, detail=f"Búfalo com ID {data.id_femea} não encontrado no banco de dados.")
            if bufalo_info['sexo'] != 'F':
                raise HTTPException(status_code=400, detail=f"Búfalo com ID {data.id_femea} não é uma fêmea.")
        
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
        logger.error(f"ERRO INTERNO: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno ao processar a predição.")

@app.post("/simular-acasalamento", response_model=SimulacaoAcasalamentoResponse, tags=["Simulação de Acasalamento"])
def simular_acasalamento(
    data: SimulacaoAcasalamentoInput,
    incluir_predicao_femea: bool = Query(True, description="Se True, inclui predição de produção da fêmea.")
):
    """Simula um acasalamento e calcula a consanguinidade da prole."""
    try:
        # Se usando Supabase, verifica se os búfalos existem
        if usando_supabase:
            macho_info = supabase_db.get_bufalo_by_id(data.id_macho)
            femea_info = supabase_db.get_bufalo_by_id(data.id_femea)
            
            if not macho_info:
                raise HTTPException(status_code=404, detail=f"Macho com ID {data.id_macho} não encontrado.")
            if not femea_info:
                raise HTTPException(status_code=404, detail=f"Fêmea com ID {data.id_femea} não encontrada.")
            if macho_info['sexo'] != 'M':
                raise HTTPException(status_code=400, detail=f"Búfalo com ID {data.id_macho} não é um macho.")
            if femea_info['sexo'] != 'F':
                raise HTTPException(status_code=400, detail=f"Búfalo com ID {data.id_femea} não é uma fêmea.")
        
        # Cria árvore genealógica
        arvore = criar_arvore_genealogica(df_historico_bufalos)
        calculador = CalculadorConsanguinidade(arvore)
        
        # Simula acasalamento
        simulacao = calculador.simular_acasalamento(data.id_macho, data.id_femea)
        
        # Predição da fêmea (opcional)
        predicao_femea = None
        if incluir_predicao_femea and model is not None:
            try:
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
            except Exception as e:
                logger.warning(f"Erro na predição da fêmea: {e}")
        
        # Monta resposta
        response_data = simulacao.copy()
        response_data['predicao_producao_femea'] = predicao_femea
        
        # Adiciona metadados
        response_data['_metadata'] = {
            'fonte_dados': 'supabase' if usando_supabase else 'csv',
            'timestamp': datetime.now().isoformat()
        }
        
        return SimulacaoAcasalamentoResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERRO NA SIMULAÇÃO: {e}")
        raise HTTPException(status_code=500, detail="Erro ao simular acasalamento.")

@app.post("/analise-genealogica", response_model=AnaliseGenealogicaResponse, tags=["Análise Genealógica"])
def analise_genealogica(data: AnaliseGenealogicaInput):
    """Analisa a genealogia de um búfalo e calcula seu coeficiente de consanguinidade."""
    try:
        # Se usando Supabase, verifica se o búfalo existe
        if usando_supabase:
            bufalo_info = supabase_db.get_bufalo_by_id(data.id_bufalo)
            if not bufalo_info:
                raise HTTPException(status_code=404, detail=f"Búfalo com ID {data.id_bufalo} não encontrado.")
        
        # Cria árvore genealógica
        arvore = criar_arvore_genealogica(df_historico_bufalos)
        calculador = CalculadorConsanguinidade(arvore)
        
        # Calcula consanguinidade
        consanguinidade = calculador.calcular_coeficiente_wright(data.id_bufalo)
        
        # Obtém ancestrais e descendentes
        ancestrais = arvore.obter_ancestrais(data.id_bufalo, max_geracoes=5)
        descendentes = arvore.obter_descendentes(data.id_bufalo, max_geracoes=3)
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERRO NA ANÁLISE GENEALÓGICA: {e}")
        raise HTTPException(status_code=500, detail="Erro ao analisar genealogia.")

@app.get("/machos-compatíveis/{femea_id}", response_model=MachosCompativeisResponse, tags=["Recomendação"])
def encontrar_machos_compatíveis(
    femea_id: int,
    max_consanguinidade: float = Query(6.25, description="Consanguinidade máxima aceitável em %")
):
    """Encontra machos compatíveis para uma fêmea baseado na consanguinidade."""
    try:
        # Se usando Supabase, verifica se a fêmea existe
        if usando_supabase:
            femea_info = supabase_db.get_bufalo_by_id(femea_id)
            if not femea_info:
                raise HTTPException(status_code=404, detail=f"Fêmea com ID {femea_id} não encontrada.")
            if femea_info['sexo'] != 'F':
                raise HTTPException(status_code=400, detail=f"Búfalo com ID {femea_id} não é uma fêmea.")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERRO NA BUSCA DE MACHOS COMPATÍVEIS: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar machos compatíveis.")

@app.get("/informacoes-femea/{femea_id}", tags=["Informações"])
def obter_informacoes_femea_endpoint(femea_id: int):
    """Obtém informações básicas de uma fêmea."""
    try:
        # Se usando Supabase, busca diretamente
        if usando_supabase:
            bufalo_info = supabase_db.get_bufalo_by_id(femea_id)
            if not bufalo_info:
                raise HTTPException(status_code=404, detail=f"Fêmea com ID {femea_id} não encontrada.")
            return bufalo_info
        
        # Senão, usa função tradicional
        info = obter_informacoes_femea(femea_id, df_historico_bufalos)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Fêmea com ID {femea_id} não encontrada.")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERRO AO OBTER INFORMAÇÕES DA FÊMEA: {e}")
        raise HTTPException(status_code=500, detail="Erro ao obter informações da fêmea.")

@app.get("/estatisticas-modelo", tags=["Modelo"])
def estatisticas_modelo():
    """Retorna estatísticas detalhadas do modelo treinado."""
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
        "feature_importance": modelo_info.get("feature_importance", []),
        "fonte_dados_atual": "Supabase (dados reais)" if usando_supabase else "CSV (dados sintéticos)",
        "dados_disponiveis": {
            "bufalos": len(df_historico_bufalos),
            "ciclos": len(df_historico_ciclos),
            "ordenhas": len(df_historico_ordenhas)
        }
    }

@app.get("/debug/dados-disponiveis", tags=["Debug"])
def debug_dados_disponiveis():
    """Debug: mostra quais dados estão disponíveis."""
    return {
        "fonte_ativa": "Supabase" if usando_supabase else "CSV",
        "supabase_conectado": supabase_db.test_connection() if supabase_db else False,
        "dados_carregados": {
            "bufalos": len(df_historico_bufalos),
            "ciclos_lactacao": len(df_historico_ciclos),
            "dados_lactacao": len(df_historico_ordenhas),
            "dados_zootecnicos": len(df_historico_zootecnicos),
            "dados_sanitarios": len(df_historico_sanitarios),
            "dados_reproducao": len(df_historico_repro)
        },
        "sample_bufalo_ids": df_historico_bufalos['id_bufalo'].head(10).tolist() if not df_historico_bufalos.empty else [],
        "range_ids": {
            "min": int(df_historico_bufalos['id_bufalo'].min()) if not df_historico_bufalos.empty else None,
            "max": int(df_historico_bufalos['id_bufalo'].max()) if not df_historico_bufalos.empty else None
        } if not df_historico_bufalos.empty else {}
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando Buffs IA v1.1.0 - Sistema de Predição Individual + Consanguinidade + Supabase")
    print("📊 Versão: 1.1.0")
    print("🔌 Supabase integrado para dados reais")
    print("📁 CSV mantido para treinamento da IA")
    print("📊 Para visualizar MLflow: mlflow ui")
    uvicorn.run(app, host="0.0.0.0", port=5001)