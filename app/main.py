from fastapi import FastAPI, HTTPException
# ... outros imports ...
from models.prediction import fazer_predicao_leite # <--- IMPORTA A FUNÇÃO

# ... (código da API e Pydantic) ...

@app.post("/prever_leite")
def prever_producao_leite(data: AcasalamentoInput):
    print("Dados recebidos do NestJS:", data.dict())

    previsao = fazer_predicao_leite(data.caracteristicas_femea.dict())

    if previsao is None:
        raise HTTPException(status_code=503, detail="Modelo de predição não está disponível.")

    return {
        "status": "sucesso",
        "previsao_potencial": {
            "producao_leite_litros_estimada": round(previsao, 2),
            "unidade": "litros/lactação",
            "observacao": "Predição baseada em um modelo de RandomForest treinado."
        }
    }