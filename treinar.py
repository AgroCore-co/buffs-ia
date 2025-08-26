# =================================================================
# ARQUIVO: treinar.py (VERSÃO 4.2 - Correção de Vazamento de Dados)
# OBJETIVO: Executar um pipeline de treinamento completo, limpo,
#           confiável e versionado com MLflow.
# =================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime

# --- CONFIGURAÇÕES ---
# Nome para registrar o modelo no MLflow Model Registry
MLFLOW_REGISTERED_MODEL_NAME = "preditor-leite-buffs"

# --- FUNÇÕES DO PIPELINE ---

def carregar_dados():
    """Carrega todos os arquivos CSV necessários."""
    print("  - Carregando arquivos CSV...")
    df_bufalos = pd.read_csv('bufalos.csv', parse_dates=['dt_nascimento'])
    df_ciclos = pd.read_csv('ciclos_lactacao.csv', parse_dates=['dt_parto'])
    df_ordenhas = pd.read_csv('dados_lactacao.csv')
    
    try:
        df_zootecnicos = pd.read_csv('dados_zootecnicos.csv')
    except FileNotFoundError:
        print("    -> AVISO: dados_zootecnicos.csv não encontrado. Será ignorado.")
        df_zootecnicos = pd.DataFrame()
        
    return df_bufalos, df_ciclos, df_ordenhas, df_zootecnicos

def processar_features(df_bufalos, df_ciclos, df_ordenhas, df_zootecnicos):
    """Executa a preparação de dados e a engenharia de features."""
    print("  - Processando e criando features...")
    
    # 1. Preparação Base
    df_producao_total = df_ordenhas.groupby('id_ciclo_lactacao')['qt_ordenha'].sum().reset_index()
    df_producao_total.rename(columns={'qt_ordenha': 'total_leite_ciclo'}, inplace=True)
    df_ciclos_prod = pd.merge(df_ciclos, df_producao_total, on='id_ciclo_lactacao')
    df_base = pd.merge(df_ciclos_prod, df_bufalos, left_on='id_bufala', right_on='id_bufalo', suffixes=('', '_mae'))

    # 2. Engenharia de Features
    df_base['idade_mae_dias'] = (df_base['dt_parto'] - df_base['dt_nascimento']).dt.days
    df_base['idade_mae_anos'] = df_base['idade_mae_dias'] / 365.25

    df_base['mes_parto'] = df_base['dt_parto'].dt.month
    df_base['estacao'] = df_base['mes_parto'] % 12 // 3 + 1

    df_base = df_base.sort_values(['id_bufala', 'dt_parto'])
    df_base['ordem_lactacao'] = df_base.groupby('id_bufala').cumcount() + 1
    
    df_base['intervalo_partos'] = df_base.groupby('id_bufala')['dt_parto'].diff().dt.days.fillna(365)

    # <<< CORREÇÃO CRÍTICA DE VAZAMENTO DE DADOS APLICADA >>>
    # O .shift(1) garante que a média usada para uma lactação seja a média de TODAS as lactações ANTERIORES.
    producao_media_mae = df_base.groupby('id_bufala')['total_leite_ciclo'].expanding().mean().shift(1).reset_index(level=0, drop=True)
    df_base['producao_media_mae'] = producao_media_mae.fillna(producao_media_mae.mean())
    
    if not df_zootecnicos.empty:
        peso_medio_pai = df_zootecnicos.groupby('id_bufalo')['peso'].mean().rename('ganho_peso_medio_pai')
        df_base = pd.merge(df_base, peso_medio_pai, left_on='id_pai', right_index=True, how='left')
        df_base['ganho_peso_medio_pai'] = df_base['ganho_peso_medio_pai'].fillna(450)
    else:
        df_base['ganho_peso_medio_pai'] = 450

    # Bloco ajustado para incluir 'id_raca' dos avós corretamente
    df_genetica_avos = df_bufalos[['id_bufalo', 'id_raca', 'potencial_genetico_leite']]
    
    df_base = pd.merge(df_base, df_genetica_avos, left_on='id_mae', right_on='id_bufalo', how='left', suffixes=('', '_avom'))
    df_base = pd.merge(df_base, df_genetica_avos, left_on='id_pai', right_on='id_bufalo', how='left', suffixes=('', '_avop'))
    
    colunas_para_remover = ['id_bufalo_avom', 'id_bufalo_avop']
    df_base.drop(columns=[col for col in colunas_para_remover if col in df_base.columns], inplace=True)
    
    df_base['potencial_genetico_avos'] = (df_base['potencial_genetico_leite_avom'].fillna(1.0) + df_base['potencial_genetico_leite_avop'].fillna(1.0)) / 2
    
    return df_base

def treinar_avaliar_modelo(df_final):
    """Treina, avalia e salva o modelo e seus artefatos, registrando tudo no MLflow."""
    print("  - Treinando e avaliando o modelo...")
    
    features_selecionadas = [
        'id_propriedade', 'producao_media_mae', 'ganho_peso_medio_pai', 'idade_mae_anos',
        'ordem_lactacao', 'estacao', 'intervalo_partos', 'potencial_genetico_avos',
        'id_raca', 'id_raca_avom'
    ]
    target = 'total_leite_ciclo'
    
    df_final.rename(columns={'id_raca_x': 'id_raca'}, inplace=True, errors='ignore')

    if 'id_raca_avom' not in df_final.columns:
        df_final['id_raca_avom'] = 0

    df_limpo = df_final.dropna(subset=[target] + features_selecionadas).copy()
    
    X = df_limpo[features_selecionadas]
    y = df_limpo[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run() as run:
        model_params = {
            "n_estimators": 150, "max_depth": 15, "min_samples_split": 5,
            "min_samples_leaf": 2, "random_state": 42, "oob_score": True
        }
        mlflow.log_params(model_params)
        
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        erro_percentual = (rmse / y.mean()) * 100 if y.mean() > 0 else 0
        
        metrics = {"rmse": rmse, "r2": r2, "oob_score": model.oob_score_, "erro_percentual": erro_percentual}
        mlflow.log_metrics(metrics)
        
        joblib.dump(model, 'modelo_leite.joblib')
        
        feature_importance = pd.DataFrame({
            'feature': features_selecionadas,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_info = {
            'versao': f"4.2-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'mlflow_run_id': run.info.run_id, 'features': features_selecionadas,
            'model_performance': metrics, 'feature_importance': feature_importance.to_dict('records'),
        }
        with open('modelo_info.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
        
        mlflow.log_artifact('modelo_info.json')

        # MELHORIA: Adicionando assinatura e exemplo para um logging mais robusto
        signature = infer_signature(X_train, predictions)
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="modelo-rf",
            signature=signature,
            input_example=X_train.head(1),
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME
        )

        print("\n" + "="*60)
        print("          RESULTADOS DO TREINAMENTO (PÓS-CORREÇÃO)")
        print("="*60)
        print(f"  -> R² (Coef. de Determinação): {r2:.4f} ({r2*100:.1f}% da variância explicada)")
        print(f"  -> RMSE (Erro Médio):          {rmse:.2f} litros ({erro_percentual:.1f}% de erro)")
        print(f"  -> OOB Score:                  {model.oob_score_:.4f}")
        print("-"*60)
        print(f"  -> Modelo salvo como 'modelo_leite.joblib'")
        print(f"  -> Informações salvas em 'modelo_info.json'")
        print(f"  -> Modelo registrado no MLflow como '{MLFLOW_REGISTERED_MODEL_NAME}'")
        print(f"  -> Para visualizar, execute: mlflow ui")
        print("="*60)

# --- EXECUÇÃO DO PIPELINE ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("        🚀 INICIANDO PIPELINE DE TREINAMENTO 🚀")
    print("="*60)
    
    df_bufalos, df_ciclos, df_ordenhas, df_zootecnicos = carregar_dados()
    df_modelo = processar_features(df_bufalos, df_ciclos, df_ordenhas, df_zootecnicos)
    treinar_avaliar_modelo(df_modelo)
    
    print("\n✅ Pipeline concluído com sucesso!")
    print("="*60 + "\n")
