# =================================================================
# ARQUIVO: gerar_dados.py (VERSÃO 1.0.0)
# OBJETIVO: Criar dataset sintético completo para o sistema de
#           predição individual + consanguinidade.
# =================================================================
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta

# --- Configurações Iniciais ---
NUM_BUFALOS = 1000
NUM_PROPRIEDADES = 4
NUM_RACAS = 4
DATA_INICIAL = date(2018, 1, 1)
DATA_FINAL = date(2024, 1, 1)

fake = Faker('pt_BR')

print("🚀 Iniciando geração de dados sintéticos para Buffs IA...")
print("📊 Sistema: Predição Individual + Consanguinidade")
print("📊 Versão: 1.0.0")

# --- Perfil de cada Propriedade ---
perfis_propriedade = {
    1: 0.85, # Baixo nível
    2: 0.95, # Baixo nível
    3: 1.15, # Bom nível
    4: 1.30  # De ponta
}

# --- Distribuição do Rebanho ---
print("Definindo distribuição do rebanho...")
total_bufalos_definidos = 20 + 40 + 110
if NUM_BUFALOS < total_bufalos_definidos:
    raise ValueError("NUM_BUFALOS é menor que a soma das propriedades com contagem fixa.")

propriedade_assignments = (
    [1] * 20 +
    [2] * 40 +
    [3] * 110 +
    [4] * (NUM_BUFALOS - total_bufalos_definidos)
)
random.shuffle(propriedade_assignments)
print(f"Distribuição final: Propriedade 1 ({propriedade_assignments.count(1)}), 2 ({propriedade_assignments.count(2)}), 3 ({propriedade_assignments.count(3)}), 4 ({propriedade_assignments.count(4)})")

# --- 1. Geração da Tabela de Búfalos ---
print("Gerando 'bufalos.csv'...")
bufalos_data = []
potencial_genetico_base = np.random.normal(1.0, 0.1, NUM_BUFALOS)

for i in range(NUM_BUFALOS):
    sexo = random.choice(['M', 'F'])
    dt_nascimento = DATA_INICIAL + timedelta(days=random.randint(0, (DATA_FINAL - DATA_INICIAL).days))
    
    id_propriedade = propriedade_assignments[i]
    
    id_pai, id_mae = None, None
    fator_propriedade = perfis_propriedade[id_propriedade]
    potencial_genetico = potencial_genetico_base[i] * fator_propriedade

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

# --- 2. Geração das Tabelas de Lactação ---
print("Gerando 'ciclos_lactacao.csv' e 'dados_lactacao.csv'...")
ciclos_data, ordenhas_data = [], []
ciclo_id_counter = 1
femeas_adultas = df_bufalos[(df_bufalos['sexo'] == 'F') & (df_bufalos['dt_nascimento'] < date(2022, 1, 1))]

for _, femea in femeas_adultas.iterrows():
    num_ciclos = random.randint(1, 3)
    dt_ultimo_parto = femea['dt_nascimento'] + timedelta(days=365 * 2)
    
    for _ in range(num_ciclos):
        if dt_ultimo_parto >= DATA_FINAL: 
            continue
            
        dt_parto = dt_ultimo_parto + timedelta(days=random.randint(330, 400))
        padrao_dias = random.choice([270, 305])
        
        # Adiciona dt_secagem_real para o sistema
        dt_secagem_real = dt_parto + timedelta(days=padrao_dias)
        
        ciclos_data.append({
            "id_ciclo_lactacao": ciclo_id_counter, 
            "id_bufala": femea['id_bufalo'], 
            "dt_parto": dt_parto,
            "dt_secagem_real": dt_secagem_real,
            "padrao_dias": padrao_dias
        })
        
        # Produção influenciada pela genética e propriedade
        producao_total_ciclo = 2500 * femea['potencial_genetico_leite'] + np.random.normal(0, 100)
        
        # Curva de lactação mais realista
        dias_lactacao = np.arange(padrao_dias)
        pico_lactacao = padrao_dias // 3  # Pico no primeiro terço
        
        # Curva de lactação com pico e declínio
        curva = np.exp(-((dias_lactacao - pico_lactacao) ** 2) / (2 * (pico_lactacao ** 2)))
        producao_diaria_normalizada = (curva / np.sum(curva)) * producao_total_ciclo if np.sum(curva) > 0 else np.zeros_like(curva)
        
        for dia, producao in enumerate(producao_diaria_normalizada):
            if producao > 0:
                ordenhas_data.append({
                    "id_lact": len(ordenhas_data) + 1, 
                    "id_ciclo_lactacao": ciclo_id_counter,
                    "qt_ordenha": max(0, round(producao + np.random.normal(0, producao * 0.1), 2)),
                    "dt_ordenha": dt_parto + timedelta(days=dia)
                })
        
        ciclo_id_counter += 1
        dt_ultimo_parto = dt_parto

df_ciclos = pd.DataFrame(ciclos_data)
df_ordenhas = pd.DataFrame(ordenhas_data)

# --- 3. Geração de Dados Zootécnicos ---
print("Gerando 'dados_zootecnicos.csv'...")
zootecnicos_data = []

for _, bufalo in df_bufalos.iterrows():
    # Gera 2 a 5 registros zootécnicos ao longo da vida do animal
    for i in range(random.randint(2, 5)):
        dias_de_vida = (DATA_FINAL - bufalo['dt_nascimento']).days
        if dias_de_vida <= 30: 
            continue
        
        dt_registro = bufalo['dt_nascimento'] + timedelta(days=random.randint(30, dias_de_vida))
        idade_anos = (dt_registro - bufalo['dt_nascimento']).days / 365.25
        
        peso = 40 + (idade_anos * 100) + np.random.normal(0, 20)
        ecc = 2.5 + (idade_anos * 0.2) + np.random.normal(0, 0.25)
        
        zootecnicos_data.append({
            "id_zootec": len(zootecnicos_data) + 1,
            "id_bufalo": bufalo['id_bufalo'],
            "peso": round(max(30, peso), 2),
            "condicao_corporal": round(max(1, min(5, ecc)), 2),
            "dt_registro": dt_registro
        })

df_zootecnicos = pd.DataFrame(zootecnicos_data)

# --- 4. Geração de Dados Sanitários ---
print("Gerando 'dados_sanitarios.csv'...")
sanitarios_data = []
doencas_comuns = ["Mastite", "Metrite", "Problema de Casco", "Laminite", "Pneumonia", "Carrapato", "Brucelose", "Leptospirose"]

for _, bufalo in df_bufalos.iterrows():
    # Simula 0 a 3 eventos sanitários na vida do animal
    if random.random() > 0.5:
        for _ in range(random.randint(1, 3)):
            dias_de_vida = (DATA_FINAL - bufalo['dt_nascimento']).days
            if dias_de_vida <= 180: 
                continue
            
            dt_aplicacao = bufalo['dt_nascimento'] + timedelta(days=random.randint(180, dias_de_vida))
            doenca = random.choice(doencas_comuns)
            
            sanitarios_data.append({
                "id_sanit": len(sanitarios_data) + 1,
                "id_bufalo": bufalo['id_bufalo'],
                "doenca": doenca,
                "medicacao": "Antibiótico" if random.random() > 0.3 else "Anti-inflamatório",
                "dt_aplicacao": dt_aplicacao
            })

df_sanitarios = pd.DataFrame(sanitarios_data)

# --- 5. Geração de Dados Reprodutivos ---
print("Gerando 'dados_reproducao.csv'...")
repro_data = []

for _, femea in femeas_adultas.iterrows():
    # Para cada fêmea, gera eventos reprodutivos
    num_eventos = random.randint(1, 4)
    
    for _ in range(num_eventos):
        # Evento de reprodução após o nascimento
        dias_de_vida = (DATA_FINAL - femea['dt_nascimento']).days
        if dias_de_vida <= 365: 
            continue
        
        dt_evento = femea['dt_nascimento'] + timedelta(days=random.randint(365, dias_de_vida))
        
        # Tipos de eventos reprodutivos
        tipos_evento = ["Inseminação", "Monta Natural", "Diagnóstico de Gestação", "Parto"]
        tipo = random.choice(tipos_evento)
        
        # Status baseado no tipo
        if tipo in ["Inseminação", "Monta Natural"]:
            status = random.choice(["Pendente", "Confirmada", "Falhou"])
        elif tipo == "Diagnóstico de Gestação":
            status = random.choice(["Positivo", "Negativo"])
        else:  # Parto
            status = "Confirmada"
        
        repro_data.append({
            "id_repro": len(repro_data) + 1,
            "id_receptora": femea['id_bufalo'],
            "tipo_evento": tipo,
            "status": status,
            "dt_evento": dt_evento,
            "observacoes": f"Evento {tipo.lower()} para fêmea {femea['id_bufalo']}"
        })

df_repro = pd.DataFrame(repro_data)

# --- 6. Salvando todos os arquivos CSV ---
print("💾 Salvando arquivos CSV...")
df_bufalos.to_csv('bufalos.csv', index=False)
df_ciclos.to_csv('ciclos_lactacao.csv', index=False)
df_ordenhas.to_csv('dados_lactacao.csv', index=False)
df_zootecnicos.to_csv('dados_zootecnicos.csv', index=False)
df_sanitarios.to_csv('dados_sanitarios.csv', index=False)
df_repro.to_csv('dados_reproducao.csv', index=False)

print("\n" + "="*60)
print("✅ ARQUIVOS CSV GERADOS COM SUCESSO!")
print("="*60)
print(f"📊 Total de búfalos: {len(df_bufalos)}")
print(f"🐄 Fêmeas adultas: {len(femeas_adultas)}")
print(f"🔄 Ciclos de lactação: {len(df_ciclos)}")
print(f"🥛 Registros de ordenha: {len(df_ordenhas)}")
print(f"⚖️ Dados zootécnicos: {len(df_zootecnicos)}")
print(f"🏥 Dados sanitários: {len(df_sanitarios)}")
print(f"👶 Dados reprodutivos: {len(df_repro)}")
print("="*60)
print("\n🚀 PRÓXIMOS PASSOS:")
print("1. Execute: python treinar_ia.py")
print("2. Execute: python -m uvicorn app.main:app --reload --port 5001")
print("3. Teste a API em: http://localhost:5001/docs")
print("="*60)
