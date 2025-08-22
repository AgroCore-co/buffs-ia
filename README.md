# BUFFS IA - Serviço de Predição Genética 🧬

Este repositório contém o serviço de Inteligência Artificial para a plataforma de manejo de búfalos **BUFFS**. A API, desenvolvida em Python com FastAPI, é responsável por abrigar os modelos de Machine Learning e fornecer endpoints para predição de potencial genético, simulação de acasalamentos e recomendação de cruzamentos.

Este serviço foi projetado para ser consumido pela API principal do BUFFS (desenvolvida em NestJS), atuando como o "cérebro" analítico do sistema.

##  Funcionalidades
d
  - **Predição de Potencial Genético:** Estima características da prole (como produção de leite, saúde, etc.) com base nos dados dos pais.
  - **Simulação de Acasalamentos:** Permite que o usuário teste virtualmente um cruzamento entre um macho e uma fêmea para visualizar o potencial da cria.
  - **Recomendação de Cruzamentos (Roadmap):** Sugere os melhores acasalamentos para um conjunto de fêmeas, otimizando para objetivos específicos como maximização da produção ou minimização da consanguinidade.

##  Tecnologias Utilizadas

  - **API:** [FastAPI](https://fastapi.tiangolo.com/) - Um moderno e performático framework web para Python.
  - **Servidor:** [Uvicorn](https://www.uvicorn.org/) - Um servidor ASGI rápido, usado para rodar a aplicação FastAPI.
  - **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) - Para o treinamento e utilização dos modelos de predição.
  - **Manipulação de Dados:** [Pandas](https://pandas.pydata.org/) - Essencial para a preparação dos datasets de treinamento.
  - **Banco de Dados:** [PostgreSQL](https://www.postgresql.org/) - Conexão via `psycopg2` para extrair os dados de treinamento.
  - **Validação de Dados:** [Pydantic](https://www.google.com/search?q=https://docs.pydantic.dev/) - Utilizado pelo FastAPI para garantir a integridade dos dados de entrada e saída da API.

##  Configuração do Ambiente

Siga os passos abaixo para configurar o ambiente de desenvolvimento local.

### 1\. Pré-requisitos

  - Python 3.9 ou superior
  - `pip` (gerenciador de pacotes do Python)

### 2\. Crie um Ambiente Virtual

É uma boa prática isolar as dependências do projeto. Na raiz do projeto, crie e ative um ambiente virtual:

```bash
# Criar o ambiente virtual
python3 -m venv .venv

# Ativar o ambiente virtual (Linux/macOS)
source .venv/bin/activate

# Ativar o ambiente virtual (Windows)
# .\.venv\Scripts\activate
```

### 3\. Instale as Dependências

Com o ambiente virtual ativo, instale todas as bibliotecas necessárias a partir do arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

##  Executando a Aplicação

Com o ambiente configurado e as dependências instaladas, inicie o servidor Uvicorn:

```bash
uvicorn app.main:app --reload --port 5001
```

  - `--reload`: O servidor reiniciará automaticamente após qualquer alteração nos arquivos. Ideal para desenvolvimento.
  - `--port 5001`: Define a porta em que a API irá rodar.

A API estará disponível em `http://127.0.0.1:5001`.

##  Endpoints da API

Você pode acessar a documentação interativa (gerada pelo Swagger UI) em `http://127.0.0.1:5001/docs`.

### Principais Endpoints

  - **`GET /`**

      - **Descrição:** Verifica o status da API.
      - **Resposta de Sucesso (200):**
        ```json
        {
          "status": "BUFFS IA API está online"
        }
        ```

  - **`POST /prever_leite`**

      - **Descrição:** Recebe as características de um macho e uma fêmea e retorna a predição da produção de leite da prole.
      - **Corpo da Requisição (JSON):**
        ```json
        {
          "caracteristicas_macho": {
            "id": 10,
            "raca": 1
          },
          "caracteristicas_femea": {
            "id": 15,
            "raca": 1
          }
        }
        ```
      - **Resposta de Sucesso (200):**
        ```json
        {
          "status": "sucesso",
          "previsao_potencial": {
            "producao_leite_litros_estimada": 2950.0,
            "unidade": "litros/lactação",
            "observacao": "Predição baseada em um modelo treinado."
          }
        }
        ```

##  Próximos Passos (Roadmap)

1.  **Extração de Dados:** Desenvolver os scripts para conectar ao banco de dados PostgreSQL e criar o dataset de treinamento inicial.
2.  **Treinamento do Modelo v1:** Treinar o primeiro modelo de regressão (`RandomForest` ou `XGBoost`) para prever a produção de leite.
3.  **Integração do Modelo:** Substituir a lógica simulada no endpoint `/prever_leite` pela chamada ao modelo treinado.
4.  **Expansão:** Criar e treinar novos modelos para prever outras características (saúde, resistência, etc.) e desenvolver o endpoint de recomendação de acasalamentos.

-----