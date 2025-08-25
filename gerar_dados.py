# =================================================================
# ARQUIVO 1: gerar_dados.py
# OBJETIVO: Criar dados fictícios e realistas para búfalos, lactações e ordenhas.
# COMO USAR: Execute este arquivo primeiro (python gerar_dados.py). Ele criará os CSVs.
# =================================================================
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

# --- Configurações Iniciais ---
NUM_BUFALOS = 500
NUM_PROPRIEDADES = 10
NUM_RACAS = 4
DATA_INICIAL = date(2018, 1, 1)
DATA_FINAL = date(2024, 1, 1)

fake = Faker('pt_BR')

print("Iniciando a geração de dados sintéticos...")

# --- 1. Geração da Tabela de Búfalos ---
print("Gerando tabela 'bufalos.csv'...")
bufalos_data = []
# O "potencial genético" é o segredo que a IA tentará aprender.
# É um valor oculto que define a qualidade intrínseca do animal.
potencial_genetico_base = np.random.normal(1.0, 0.1, NUM_BUFALOS)

for i in range(NUM_BUFALOS):
    sexo = random.choice(['M', 'F'])
    dt_nascimento = DATA_INICIAL + timedelta(days=random.randint(0, (DATA_FINAL - DATA_INICIAL).days))
    
    # Lógica para garantir que os pais sejam mais velhos
    id_pai = None
    id_mae = None
    potencial_genetico = potencial_genetico_base[i]

    # A partir do 20º búfalo, começamos a atribuir pais
    if i > 20:
        # Busca por mães e pais elegíveis (mais velhos)
        pais_possiveis = [b for b in bufalos_data if b['dt_nascimento'] < dt_nascimento]
        maes_elegiveis = [p for p in pais_possiveis if p['sexo'] == 'F']
        pais_elegiveis = [p for p in pais_possiveis if p['sexo'] == 'M']
        
        if maes_elegiveis and pais_elegiveis:
            mae = random.choice(maes_elegiveis)
            pai = random.choice(pais_elegiveis)
            id_mae = mae['id_bufalo']
            id_pai = pai['id_bufalo']
            
            # A genética da cria é a média dos pais + uma pequena variação aleatória
            potencial_genetico = (mae['potencial_genetico_leite'] + pai['potencial_genetico_leite']) / 2 + np.random.normal(0, 0.05)

    bufalos_data.append({
        "id_bufalo": i + 1,
        "nome": fake.first_name(),
        "sexo": sexo,
        "dt_nascimento": dt_nascimento,
        "id_raca": random.randint(1, NUM_RACAS),
        "id_propriedade": random.randint(1, NUM_PROPRIEDADES),
        "id_pai": id_pai,
        "id_mae": id_mae,
        "potencial_genetico_leite": potencial_genetico # Esta coluna NÃO será usada no treino, apenas para gerar o alvo
    })

df_bufalos = pd.DataFrame(bufalos_data)

# --- 2. Geração das Tabelas de Ciclo e Dados de Lactação ---
print("Gerando tabelas 'ciclos_lactacao.csv' e 'dados_lactacao.csv'...")
ciclos_data = []
ordenhas_data = []
ciclo_id_counter = 1

femeas_adultas = df_bufalos[(df_bufalos['sexo'] == 'F') & (df_bufalos['dt_nascimento'] < date(2022, 1, 1))]

for _, femea in femeas_adultas.iterrows():
    # Cada fêmea pode ter de 1 a 3 ciclos de lactação
    num_ciclos = random.randint(1, 3)
    dt_ultimo_parto = femea['dt_nascimento'] + timedelta(days=365 * 2) # Primeiro parto com ~2 anos

    for _ in range(num_ciclos):
        if dt_ultimo_parto >= DATA_FINAL:
            continue

        dt_parto = dt_ultimo_parto + timedelta(days=random.randint(330, 400))
        padrao_dias = random.choice([270, 305])
        
        ciclos_data.append({
            "id_ciclo_lactacao": ciclo_id_counter,
            "id_bufala": femea['id_bufalo'],
            "dt_parto": dt_parto
        })
        
        # Lógica para gerar a produção de leite baseada na genética
        producao_total_ciclo = 2500 * femea['potencial_genetico_leite'] + np.random.normal(0, 100)
        
        # Simula uma curva de lactação (pico no início, depois declínio)
        dias_lactacao = np.arange(padrao_dias)
        # Função matemática para a curva: a * x * exp(-b * x)
        curva = dias_lactacao * np.exp(-dias_lactacao / 100.0)
        producao_diaria_normalizada = (curva / np.sum(curva)) * producao_total_ciclo
        
        for dia, producao in enumerate(producao_diaria_normalizada):
            ordenhas_data.append({
                "id_lact": len(ordenhas_data) + 1,
                "id_ciclo_lactacao": ciclo_id_counter,
                "qt_ordenha": max(0, producao + np.random.normal(0, 0.5)), # Adiciona ruído
                "dt_ordenha": dt_parto + timedelta(days=dia)
            })
        
        ciclo_id_counter += 1
        dt_ultimo_parto = dt_parto

df_ciclos = pd.DataFrame(ciclos_data)
df_ordenhas = pd.DataFrame(ordenhas_data)

# --- 3. Salvando os arquivos CSV ---
df_bufalos.to_csv('bufalos.csv', index=False)
df_ciclos.to_csv('ciclos_lactacao.csv', index=False)
df_ordenhas.to_csv('dados_lactacao.csv', index=False)

print("\nArquivos CSV gerados com sucesso:")
print(f"- bufalos.csv ({len(df_bufalos)} registros)")
print(f"- ciclos_lactacao.csv ({len(df_ciclos)} registros)")
print(f"- dados_lactacao.csv ({len(df_ordenhas)} registros)")

