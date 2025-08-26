# 🐃 Buffs IA - Sistema de Predição Individual e Consanguinidade

## 📊 Versão 1.0.0

Sistema inteligente para **predição individual de produção de leite** e **análise de consanguinidade** em búfalos, desenvolvido para otimizar o manejo reprodutivo e produtivo do rebanho.

## 🎯 **Funcionalidades Principais**

### 1. **Predição Individual de Produção de Leite**
- 🥛 **Predição personalizada** para cada fêmea em seu próximo ciclo
- 📈 **Features avançadas**: histórico produtivo, saúde, reprodução, genética
- 🎯 **Meta de precisão**: R² > 0.70
- 📊 **Classificação de potencial**: Alto, Bom, Médio, Baixo

### 2. **Análise de Consanguinidade**
- 🧬 **Coeficiente de Wright** para cálculo preciso
- 🌳 **Árvores genealógicas** interativas
- ⚠️ **Classificação de risco**: Baixo, Médio, Alto
- 🔍 **Análise de ancestrais** até 5 gerações

### 3. **Simulação de Acasalamentos**
- 💑 **Simulação completa** de acasalamentos
- 📊 **Cálculo de consanguinidade** da prole
- 🎯 **Recomendações inteligentes** baseadas em risco genético
- 🚫 **Filtros automáticos** para evitar acasalamentos de alto risco

### 4. **Recomendação de Machos Compatíveis**
- 🔍 **Busca inteligente** de machos compatíveis
- 📊 **Ranking por compatibilidade** genética
- ⚡ **Resposta em < 1 segundo**
- 🎯 **Limites configuráveis** de consanguinidade

## 🏗️ **Arquitetura do Sistema**

```
📁 buffs-ia/
├── 🚀 app/main.py                 # API FastAPI principal
├── 🧠 app/models/
│   ├── prediction.py              # Módulo de predição individual
│   └── genealogia.py             # Módulo de consanguinidade
├── 📊 treinar_ia.py              # Script de treinamento da IA
├── 📈 gerar_dados.py             # Geração de dados sintéticos
├── 📋 requirements.txt            # Dependências Python
└── 📚 README.md                   # Este arquivo
```

## 🚀 **Como Usar**

### **Passo 1: Preparação dos Dados**
```bash
# Gera dataset sintético completo
python gerar_dados.py
```

**Arquivos gerados:**
- `bufalos.csv` - Dados dos búfalos + genealogia
- `ciclos_lactacao.csv` - Ciclos de lactação
- `dados_lactacao.csv` - Produção diária de leite
- `dados_zootecnicos.csv` - Peso, ECC, etc.
- `dados_sanitarios.csv` - Histórico de saúde
- `dados_reproducao.csv` - Eventos reprodutivos

### **Passo 2: Treinamento da IA**
```bash
# Treina o modelo de predição individual
python treinar_ia.py
```

**Resultados esperados:**
- ✅ R² > 0.70 (meta de precisão)
- 📊 Modelo salvo como `modelo_producao_individual.joblib`
- 🔍 Informações salvas em `modelo_producao_individual_info.json`
- 📈 Modelo registrado no MLflow

### **Passo 3: Execução da API**
```bash
# Inicia a API em modo desenvolvimento
python -m uvicorn app.main:app --reload --port 5001
```

**Acesso:**
- 🌐 **API**: http://localhost:5001
- 📚 **Documentação**: http://localhost:5001/docs
- 🔍 **Testes**: http://localhost:5001/redoc

## 📡 **Endpoints da API**

### **Predição Individual**
```http
POST /predicao-individual
{
  "id_femea": 123
}
```

### **Simulação de Acasalamento**
```http
POST /simular-acasalamento
{
  "id_macho": 456,
  "id_femea": 123
}
```

### **Análise Genealógica**
```http
POST /analise-genealogica
{
  "id_bufalo": 123
}
```

### **Machos Compatíveis**
```http
GET /machos-compatíveis/123?max_consanguinidade=6.25
```

### **Informações da Fêmea**
```http
GET /informacoes-femea/123
```

### **Estatísticas do Modelo**
```http
GET /estatisticas-modelo
```

## 🔧 **Configurações**

### **Limites de Consanguinidade**
- 🟢 **Baixo risco**: < 3.125%
- 🟡 **Médio risco**: 3.125% - 6.25%
- 🔴 **Alto risco**: > 6.25%

### **Features do Modelo**
1. `id_propriedade` - Perfil da propriedade
2. `idade_mae_anos` - Idade da fêmea
3. `ordem_lactacao` - Ordem do ciclo
4. `estacao` - Estação do ano
5. `intervalo_partos` - Intervalo entre partos
6. `producao_media_historica` - Histórico produtivo
7. `id_raca` - Raça do animal
8. `contagem_tratamentos` - Eventos de saúde
9. `flag_doenca_grave` - Doenças graves
10. `ecc_medio_ciclo` - Condição corporal
11. `idade_primeiro_parto_dias` - Idade no primeiro parto
12. `dias_em_aberto` - Período pós-parto
13. `potencial_genetico_mae` - Potencial genético

## 📊 **Métricas de Sucesso**

### **Modelo de Predição**
- 🎯 **R² > 0.70** (70% da variância explicada)
- 📉 **RMSE < 200 litros** (erro médio)
- 🔍 **OOB Score > 0.65** (validação cruzada)

### **Sistema de Consanguinidade**
- ✅ **100% de precisão** nos cálculos
- ⚡ **Resposta < 1 segundo** para simulações
- 🎯 **Filtros automáticos** para alto risco

### **Adoção e Validação**
- 🏭 **Piloto em fazendas** selecionadas
- 📉 **Redução de acasalamentos** de alto risco
- 👨‍🌾 **Aprovação de especialistas** veterinários

## 🛠️ **Tecnologias Utilizadas**

- **Python 3.8+** - Linguagem principal
- **FastAPI** - Framework da API
- **scikit-learn** - Machine Learning
- **RandomForest** - Algoritmo de predição
- **MLflow** - Experiment tracking e model registry
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Pydantic** - Validação de dados

## 🔮 **Próximos Passos**

### **Fase 1: Integração com Banco de Dados**
- 🗄️ **Supabase/PostgreSQL** para dados reais
- 🔄 **Queries SQL/ORM** otimizadas
- 📊 **Sincronização automática** de dados

### **Fase 2: Deploy e Containerização**
- 🐳 **Docker** para containerização
- ☁️ **Deploy em nuvem** (AWS)
- 🔄 **CI/CD pipeline** automatizado

### **Fase 3: Validação**
- 🧪 **Testes com especialistas** veterinários
- 📊 **Validação em campo** com dados reais

## 📄 **Licença**

Este projeto está licenciado sob a [MIT License](LICENSE).

---

**🐃 Buffs IA - Transformando o manejo de búfalos com inteligência artificial! 🚀**
