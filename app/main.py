from fastapi import FastAPI
from pydantic import BaseModel

# --- DTOs (Data Transfer Objects) com Pydantic ---
# Define a estrutura dos dados que esperamos receber

class CaracteristicasBufalo(BaseModel):
    id: int
    raca: int
    # Adicione outros campos que virão do NestJS no futuro

class AcasalamentoInput(BaseModel):
    caracteristicas_macho: CaracteristicasBufalo
    caracteristicas_femea: CaracteristicasBufalo

# --- Inicialização da API ---
app = FastAPI(
    title="BUFFS IA API",
    description="Serviço de predição para o manejo reprodutivo de búfalos.",
    version="0.1.0"
)

# --- Endpoints da API ---
@app.get("/")
def read_root():
    return {"status": "BUFFS IA API está online"}

@app.post("/prever_leite")
def prever_producao_leite(data: AcasalamentoInput):
    """
    Recebe os dados de um macho e uma fêmea e retorna uma predição
    da produção de leite da futura prole.
    """
    print("Dados recebidos do NestJS:", data.dict())

    # --- LÓGICA DA IA (VERSÃO SIMULADA) ---
    # Aqui entrará a chamada para o seu modelo treinado.
    # Por enquanto, vamos retornar um valor calculado simples.

    # Exemplo de lógica simples: pega a raça da fêmea e adiciona um fator aleatório
    previsao = 2800 + (data.caracteristicas_femea.raca * 150) + (100 * (1- 0.5))

    return {
        "status": "sucesso",
        "previsao_potencial": {
            "producao_leite_litros_estimada": round(previsao, 2),
            "unidade": "litros/lactação",
            "observacao": "Predição baseada em um modelo de exemplo."
        }
    }