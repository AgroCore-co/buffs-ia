# BUFFS IA - Serviço de Predição 🧬

API em FastAPI para modelos de ML que estimam produção de leite por ciclo e simulam acasalamentos. Inclui geração de dados sintéticos, pipeline de treinamento com MLflow e endpoint de predição.

## Funcionalidades
- **Predição por ciclo:** Regressão da produção total de leite por ciclo de lactação.
- **Engenharia de features do rebanho:** Idade/ordem de lactação, sazonalidade, histórico próprio, genética de avós, saúde e reprodução.
- **Simulação de acasalamentos:** Endpoint para estimar potencial com base na fêmea (com suporte a contexto da propriedade).

## Tecnologias
- **API:** FastAPI + Uvicorn
- **ML:** scikit-learn, MLflow (registro de modelos/artefatos)
- **Dados:** pandas, numpy

## Ambiente
### Pré-requisitos
- Python 3.10+

### Virtualenv e dependências
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Dados
Você pode usar dados próprios (CSV) ou gerar dados sintéticos.

- Arquivos esperados pelo pipeline de treino:
  - `bufalos.csv`
  - `ciclos_lactacao.csv`
  - `dados_lactacao.csv`
  - `dados_zootecnicos.csv` (opcional, para ECC/peso)
  - `dados_sanitarios.csv` (opcional, saúde)
  - `dados_reproducao.csv` (opcional, reprodução)

### Gerar dados sintéticos
```bash
python gerar_dados.py
```
Gera: `bufalos.csv`, `ciclos_lactacao.csv`, `dados_lactacao.csv`, `dados_zootecnicos.csv`, `dados_sanitarios.csv`.

## Treinamento
Execute o pipeline completo com logging no MLflow:
```bash
python treinar.py
```
Saídas:
- `modelo_leite.joblib` (modelo RandomForestRegressor)
- `modelo_info.json` (features, métricas, importâncias)
- Registro no MLflow Model Registry (`preditor-leite-buffs`)

Para visualizar o MLflow UI:
```bash
mlflow ui
```

### Features criadas (principal)
- Base: `idade_mae_anos`, `ordem_lactacao`, `estacao`, `intervalo_partos`, `producao_media_mae`, `ganho_peso_medio_pai`, `potencial_genetico_avos`, `id_raca`, `id_raca_avom`.
- Saúde (usa janela [dt_parto, dt_secagem_real ou dt_parto+padrao_dias]):
  - `contagem_tratamentos` (COUNT em `dados_sanitarios`)
  - `flag_doenca_grave` (palavras-chave: mastite, metrite, podal, ...)
  - `ecc_medio_ciclo` (AVG `condicao_corporal` em `dados_zootecnicos`)
- Reprodução:
  - `idade_primeiro_parto_dias`
  - `dias_em_aberto` (até primeira concepção confirmada após o parto; requer `dados_reproducao.csv`)

Observação: quando arquivos opcionais estão ausentes, o pipeline aplica defaults seguros (ex.: ECC=3.0, contagem=0) para evitar NaNs.

## API
Inicie a API:
```bash
uvicorn app.main:app --reload --port 5001
```
Swagger: `http://127.0.0.1:5001/docs`

### Endpoint principal
- `POST /prever-acasalamento`

Exemplo:
```bash
curl -X POST "http://127.0.0.1:5001/prever-acasalamento?incluir_detalhes_pais=true" \
  -H "Content-Type: application/json" \
  -d '{"id_macho": 1, "id_femea": 1}'
```

Resposta (campos principais):
- `producao_estimada_litros`: previsão do modelo
- `classificacao_potencial`: comparação com média da propriedade
- `contexto_propriedade`: id, média local e diferença percentual
- `detalhes_pais` (opcional): info bruta dos pais se solicitado

Notas:
- A predição usa as features definidas em `modelo_info.json`. Hoje, o conjunto de features é centrado na fêmea e contexto do rebanho; o `id_macho` é retornado apenas como metadado quando solicitado.
- As médias por propriedade são calculadas dos CSVs carregados no startup da API.

## Estrutura
- `gerar_dados.py`: gera CSVs sintéticos
- `treinar.py`: pipeline de treinamento + MLflow
- `app/main.py`: API FastAPI e endpoint `/prever-acasalamento`
- `app/models/prediction.py`: utilidades de predição (legado/suporte)
