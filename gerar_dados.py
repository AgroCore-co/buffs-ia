# =================================================================
# ARQUIVO 1: gerar_dados.py (VERSÃO 2 - CONTEXTUAL)
# OBJETIVO: Criar dados fictícios onde cada propriedade tem sua própria realidade de produção.
# =================================================================
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

# --- Configurações Iniciais ---
NUM_BUFALOS = 1000
NUM_PROPRIEDADES = 10
NUM_RACAS = 4
DATA_INICIAL = date(2018, 1, 1)
DATA_FINAL = date(2024, 1, 1)

fake = Faker('pt_BR')

print("Iniciando a geração de dados sintéticos CONTEXTUAIS...")

# --- NOVO: Perfil de cada Propriedade ---
# Cada propriedade terá um fator que multiplica a produção.
# A propriedade 3 será a do "Marco Aurélio", com animais de ponta.
perfis_propriedade = {i: random.uniform(0.85, 1.15) for i in range(1, NUM_PROPRIEDADES + 1)}
perfis_propriedade[3] = 1.4 # Fator de alta produção
print(f"Perfis de produção das propriedades: {perfis_propriedade}")

# --- 1. Geração da Tabela de Búfalos ---
print("Gerando tabela 'bufalos.csv'...")
bufalos_data = []
potencial_genetico_base = np.random.normal(1.0, 0.1, NUM_BUFALOS)

for i in range(NUM_BUFALOS):
    sexo = random.choice(['M', 'F'])
    dt_nascimento = DATA_INICIAL + timedelta(days=random.randint(0, (DATA_FINAL - DATA_INICIAL).days))
    id_propriedade = random.randint(1, NUM_PROPRIEDADES)
    
    id_pai, id_mae = None, None
    fator_propriedade = perfis_propriedade[id_propriedade]
    potencial_genetico = potencial_genetico_base[i] * fator_propriedade # Genética influenciada pela propriedade

    if i > 20:
        pais_possiveis = [b for b in bufalos_data if b['dt_nascimento'] < dt_nascimento]
        maes_elegiveis = [p for p in pais_possiveis if p['sexo'] == 'F']
        pais_elegiveis = [p for p in pais_possiveis if p['sexo'] == 'M']
        
        if maes_elegiveis and pais_elegiveis:
            mae = random.choice(maes_elegiveis)
            pai = random.choice(pais_elegiveis)
            id_mae, id_pai = mae['id_bufalo'], pai['id_bufalo']
            potencial_genetico = (mae['potencial_genetico_leite'] + pai['potencial_genetico_leite']) / 2 + np.random.normal(0, 0.05)

    bufalos_data.append({
        "id_bufalo": i + 1, "sexo": sexo, "dt_nascimento": dt_nascimento,
        "id_raca": random.randint(1, NUM_RACAS), "id_propriedade": id_propriedade,
        "id_pai": id_pai, "id_mae": id_mae,
        "potencial_genetico_leite": potencial_genetico
    })

df_bufalos = pd.DataFrame(bufalos_data)

# --- 2. Geração das Tabelas de Ciclo e Dados de Lactação ---
print("Gerando tabelas de lactação...")
ciclos_data, ordenhas_data = [], []
ciclo_id_counter = 1

femeas_adultas = df_bufalos[(df_bufalos['sexo'] == 'F') & (df_bufalos['dt_nascimento'] < date(2022, 1, 1))]

for _, femea in femeas_adultas.iterrows():
    num_ciclos = random.randint(1, 3)
    dt_ultimo_parto = femea['dt_nascimento'] + timedelta(days=365 * 2)

    for _ in range(num_ciclos):
        if dt_ultimo_parto >= DATA_FINAL: continue
        dt_parto = dt_ultimo_parto + timedelta(days=random.randint(330, 400))
        padrao_dias = random.choice([270, 305])
        
        ciclos_data.append({"id_ciclo_lactacao": ciclo_id_counter, "id_bufala": femea['id_bufalo'], "dt_parto": dt_parto})
        
        producao_total_ciclo = 2500 * femea['potencial_genetico_leite'] + np.random.normal(0, 100)
        
        dias_lactacao = np.arange(padrao_dias)
        curva = dias_lactacao * np.exp(-dias_lactacao / 100.0)
        producao_diaria_normalizada = (curva / np.sum(curva)) * producao_total_ciclo
        
        for dia, producao in enumerate(producao_diaria_normalizada):
            ordenhas_data.append({
                "id_lact": len(ordenhas_data) + 1, "id_ciclo_lactacao": ciclo_id_counter,
                "qt_ordenha": max(0, producao + np.random.normal(0, 0.5)),
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

print("\nArquivos CSV contextuais gerados com sucesso!")
