# =================================================================
# ARQUIVO: prediction.py (VERSÃO 1.0.0)
# OBJETIVO: Módulo para predição individual de produção de leite
#           baseado no histórico da fêmea.
# =================================================================
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any, Optional

# Carrega o modelo treinado
try:
    model = joblib.load('modelo_producao_individual.joblib')
    print("✅ Modelo de predição individual carregado com sucesso!")
except FileNotFoundError:
    print("❌ AVISO: modelo_producao_individual.joblib não encontrado.")
    model = None

def preparar_features_femea(
    id_femea: int,
    df_bufalos: pd.DataFrame,
    df_ciclos: pd.DataFrame,
    df_ordenhas: pd.DataFrame,
    df_zootecnicos: pd.DataFrame,
    df_sanitarios: pd.DataFrame,
    df_repro: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepara features para uma fêmea específica baseado em seu histórico.
    """
    # Filtra dados da fêmea
    femea = df_bufalos[df_bufalos['id_bufalo'] == id_femea]
    if femea.empty:
        raise ValueError(f"Fêmea com ID {id_femea} não encontrada.")
    
    if femea.iloc[0]['sexo'] != 'F':
        raise ValueError(f"Animal com ID {id_femea} não é uma fêmea.")
    
    # Obtém ciclos da fêmea
    ciclos_femea = df_ciclos[df_ciclos['id_bufala'] == id_femea]
    if ciclos_femea.empty:
        raise ValueError(f"Fêmea {id_femea} não possui ciclos de lactação.")
    
    # Ordena por data de parto (mais recente primeiro)
    ciclos_femea = ciclos_femea.sort_values('dt_parto', ascending=False)
    
    # Calcula produção total por ciclo
    producao_por_ciclo = df_ordenhas.groupby('id_ciclo_lactacao')['qt_ordenha'].sum().reset_index()
    producao_por_ciclo.rename(columns={'qt_ordenha': 'total_leite_ciclo'}, inplace=True)
    
    # Merge com ciclos
    ciclos_com_producao = pd.merge(ciclos_femea, producao_por_ciclo, on='id_ciclo_lactacao', how='left')
    
    # Features demográficas e reprodutivas
    ciclos_com_producao['idade_mae_dias'] = (ciclos_com_producao['dt_parto'] - femea.iloc[0]['dt_nascimento']).dt.days
    ciclos_com_producao['idade_mae_anos'] = ciclos_com_producao['idade_mae_dias'] / 365.25
    
    ciclos_com_producao['mes_parto'] = ciclos_com_producao['dt_parto'].dt.month
    ciclos_com_producao['estacao'] = ciclos_com_producao['mes_parto'] % 12 // 3 + 1
    
    # Ordem de lactação
    ciclos_com_producao = ciclos_com_producao.sort_values('dt_parto')
    ciclos_com_producao['ordem_lactacao'] = range(1, len(ciclos_com_producao) + 1)
    
    # Intervalo entre partos
    ciclos_com_producao['intervalo_partos'] = ciclos_com_producao['dt_parto'].diff().dt.days.fillna(365)
    
    # Produção média histórica (sem vazamento de dados)
    ciclos_com_producao['producao_media_historica'] = ciclos_com_producao['total_leite_ciclo'].expanding().mean().shift(1)
    ciclos_com_producao['producao_media_historica'] = ciclos_com_producao['producao_media_historica'].fillna(
        ciclos_com_producao['total_leite_ciclo'].mean()
    )
    
    # Features de saúde por ciclo
    ciclos_com_producao['contagem_tratamentos'] = 0
    ciclos_com_producao['flag_doenca_grave'] = 0
    ciclos_com_producao['ecc_medio_ciclo'] = 3.0
    
    # Calcula saúde por ciclo se houver dados
    if not df_sanitarios.empty and 'doenca' in df_sanitarios.columns:
        df_sanitarios['doenca'] = df_sanitarios['doenca'].astype(str).str.lower()
        palavras_chave = ['mastite', 'metrite', 'podal', 'laminite', 'brucelose', 'leptospirose']
        
        for idx, ciclo in ciclos_com_producao.iterrows():
            ciclo_id = ciclo['id_ciclo_lactacao']
            inicio = ciclo['dt_parto']
            
            # Determina fim do ciclo
            if pd.notna(ciclo.get('dt_secagem_real')):
                fim = ciclo['dt_secagem_real']
            elif 'padrao_dias' in ciclo:
                fim = inicio + pd.Timedelta(days=ciclo['padrao_dias'])
            else:
                fim = inicio + pd.Timedelta(days=305)
            
            # Conta tratamentos no ciclo
            tratamentos = df_sanitarios[
                (df_sanitarios['id_bufalo'] == id_femea) &
                (df_sanitarios['dt_aplicacao'] >= inicio) &
                (df_sanitarios['dt_aplicacao'] <= fim)
            ]
            
            ciclos_com_producao.loc[idx, 'contagem_tratamentos'] = len(tratamentos)
            
            # Verifica doenças graves
            if not tratamentos.empty:
                has_grave = tratamentos['doenca'].apply(
                    lambda x: any(k in x for k in palavras_chave)
                ).any()
                ciclos_com_producao.loc[idx, 'flag_doenca_grave'] = 1 if has_grave else 0
    
    # Calcula ECC médio por ciclo se houver dados
    if not df_zootecnicos.empty and 'condicao_corporal' in df_zootecnicos.columns:
        for idx, ciclo in ciclos_com_producao.iterrows():
            inicio = ciclo['dt_parto']
            
            if pd.notna(ciclo.get('dt_secagem_real')):
                fim = ciclo['dt_secagem_real']
            elif 'padrao_dias' in ciclo:
                fim = inicio + pd.Timedelta(days=ciclo['padrao_dias'])
            else:
                fim = inicio + pd.Timedelta(days=305)
            
            registros_ecc = df_zootecnicos[
                (df_zootecnicos['id_bufalo'] == id_femea) &
                (df_zootecnicos['dt_registro'] >= inicio) &
                (df_zootecnicos['dt_registro'] <= fim)
            ]
            
            if not registros_ecc.empty:
                ciclos_com_producao.loc[idx, 'ecc_medio_ciclo'] = registros_ecc['condicao_corporal'].mean()
    
    # Features reprodutivas
    ciclos_com_producao['idade_primeiro_parto_dias'] = ciclos_com_producao['idade_mae_dias'].iloc[0]
    
    # Dias em aberto (do parto até primeira concepção)
    ciclos_com_producao['dias_em_aberto'] = np.nan
    if not df_repro.empty:
        for idx, ciclo in ciclos_com_producao.iterrows():
            if ciclo['ordem_lactacao'] == 1:
                continue
                
            dt_parto = ciclo['dt_parto']
            eventos_futuros = df_repro[
                (df_repro['id_receptora'] == id_femea) &
                (df_repro['dt_evento'] > dt_parto) &
                (df_repro.get('status', '').astype(str).str.lower() == 'confirmada')
            ]
            
            if not eventos_futuros.empty:
                concepcao = eventos_futuros['dt_evento'].min()
                ciclos_com_producao.loc[idx, 'dias_em_aberto'] = (concepcao - dt_parto).days
    
    # Features genéticas
    ciclos_com_producao['id_raca'] = femea.iloc[0].get('id_raca', 0)
    ciclos_com_producao['potencial_genetico_mae'] = femea.iloc[0].get('potencial_genetico_leite', 1.0)
    ciclos_com_producao['id_propriedade'] = femea.iloc[0].get('id_propriedade', 0)
    
    return ciclos_com_producao

def fazer_predicao_producao_individual(
    id_femea: int,
    df_bufalos: pd.DataFrame,
    df_ciclos: pd.DataFrame,
    df_ordenhas: pd.DataFrame,
    df_zootecnicos: pd.DataFrame,
    df_sanitarios: pd.DataFrame,
    df_repro: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Faz predição de produção individual para uma fêmea.
    """
    if model is None:
        print("❌ Modelo não disponível para predição.")
        return None
    
    try:
        # Prepara features
        df_features = preparar_features_femea(
            id_femea, df_bufalos, df_ciclos, df_ordenhas, 
            df_zootecnicos, df_sanitarios, df_repro
        )
        
        # Seleciona features para predição
        features_selecionadas = [
            'id_propriedade', 'idade_mae_anos', 'ordem_lactacao', 'estacao',
            'intervalo_partos', 'producao_media_historica', 'id_raca',
            'contagem_tratamentos', 'flag_doenca_grave', 'ecc_medio_ciclo',
            'idade_primeiro_parto_dias', 'dias_em_aberto', 'potencial_genetico_mae'
        ]
        
        # Verifica se todas as features estão disponíveis
        for feature in features_selecionadas:
            if feature not in df_features.columns:
                print(f"❌ Feature '{feature}' não encontrada.")
                return None
        
        # Prepara dados para predição (último ciclo como base)
        ultimo_ciclo = df_features.iloc[-1]
        X_pred = ultimo_ciclo[features_selecionadas].values.reshape(1, -1)
        
        # Faz predição
        predicao_litros = model.predict(X_pred)[0]
        
        # Classifica potencial
        if predicao_litros > 3000:
            classificacao = "Alto Potencial"
        elif predicao_litros > 2500:
            classificacao = "Bom Potencial"
        elif predicao_litros > 2000:
            classificacao = "Potencial Médio"
        else:
            classificacao = "Potencial Baixo"
        
        # Calcula percentual vs média da propriedade
        producao_media_prop = df_features[df_features['id_propriedade'] == ultimo_ciclo['id_propriedade']]['total_leite_ciclo'].mean()
        percentual_vs_media = ((predicao_litros - producao_media_prop) / producao_media_prop) * 100 if producao_media_prop > 0 else 0
        
        return {
            "id_femea": id_femea,
            "predicao_litros": round(predicao_litros, 2),
            "classificacao_potencial": classificacao,
            "percentual_vs_media": round(percentual_vs_media, 1),
            "producao_media_propriedade": round(producao_media_prop, 2),
            "id_propriedade": int(ultimo_ciclo['id_propriedade']),
            "features_utilizadas": features_selecionadas,
            "data_predicao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"❌ Erro na predição individual: {e}")
        return None

def obter_informacoes_femea(
    id_femea: int,
    df_bufalos: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Obtém informações básicas de uma fêmea.
    """
    try:
        femea = df_bufalos[df_bufalos['id_bufalo'] == id_femea]
        if femea.empty:
            return None
        
        femea_data = femea.iloc[0]
        
        # Trata a data de nascimento corretamente
        dt_nascimento = femea_data['dt_nascimento']
        if isinstance(dt_nascimento, str):
            dt_nascimento_str = dt_nascimento
        else:
            dt_nascimento_str = dt_nascimento.strftime("%Y-%m-%d")
        
        return {
            "id_bufalo": int(femea_data['id_bufalo']),
            "sexo": femea_data['sexo'],
            "dt_nascimento": dt_nascimento_str,
            "id_raca": int(femea_data['id_raca']),
            "id_propriedade": int(femea_data['id_propriedade']),
            "id_pai": int(femea_data['id_pai']) if pd.notna(femea_data['id_pai']) else None,
            "id_mae": int(femea_data['id_mae']) if pd.notna(femea_data['id_mae']) else None,
            "potencial_genetico_leite": float(femea_data['potencial_genetico_leite'])
        }
        
    except Exception as e:
        print(f"❌ Erro ao obter informações da fêmea: {e}")
        return None