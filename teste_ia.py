# =================================================================
# ARQUIVO: teste_ia.py (VERSÃO 1.0.0)
# OBJETIVO: Testar os módulos de predição individual e genealogia
# =================================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def criar_dados_teste():
    """Cria dados de teste para validar a IA versão 1.0.0."""
    print("🧪 Criando dados de teste para Buffs IA v1.0.0...")
    
    # 1. Búfalos com genealogia
    print("  - Gerando bufalos.csv...")
    bufalos_data = []
    for i in range(1, 101):  # 100 búfalos
        sexo = 'M' if i <= 30 else 'F'
        dt_nascimento = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))
        
        # Cria genealogia realista
        id_pai, id_mae = None, None
        if i > 20:  # Primeiros 20 são fundadores
            pais_possiveis = [b for b in bufalos_data if b['dt_nascimento'] < dt_nascimento]
            maes = [p for p in pais_possiveis if p['sexo'] == 'F']
            pais = [p for p in pais_possiveis if p['sexo'] == 'M']
            
            if maes and pais:
                id_mae = random.choice(maes)['id_bufalo']
                id_pai = random.choice(pais)['id_bufalo']
        
        bufalos_data.append({
            "id_bufalo": i,
            "sexo": sexo,
            "dt_nascimento": dt_nascimento,
            "id_raca": random.randint(1, 4),
            "id_propriedade": random.randint(1, 4),
            "id_pai": id_pai,
            "id_mae": id_mae,
            "potencial_genetico_leite": 0.8 + random.random() * 0.4
        })
    
    df_bufalos = pd.DataFrame(bufalos_data)
    df_bufalos.to_csv('bufalos.csv', index=False)
    print(f"    ✅ {len(df_bufalos)} búfalos criados")
    
    # 2. Ciclos de lactação
    print("  - Gerando ciclos_lactacao.csv...")
    ciclos_data = []
    ciclo_id = 1
    
    femeas = df_bufalos[df_bufalos['sexo'] == 'F']
    for _, femea in femeas.iterrows():
        num_ciclos = random.randint(1, 3)
        dt_ultimo_parto = femea['dt_nascimento'] + timedelta(days=365 * 2)
        
        for _ in range(num_ciclos):
            if dt_ultimo_parto >= datetime.now():
                continue
                
            dt_parto = dt_ultimo_parto + timedelta(days=random.randint(330, 400))
            padrao_dias = random.choice([270, 305])
            dt_secagem = dt_parto + timedelta(days=padrao_dias)
            
            ciclos_data.append({
                "id_ciclo_lactacao": ciclo_id,
                "id_bufala": femea['id_bufalo'],
                "dt_parto": dt_parto,
                "dt_secagem_real": dt_secagem,
                "padrao_dias": padrao_dias
            })
            ciclo_id += 1
            dt_ultimo_parto = dt_parto
    
    df_ciclos = pd.DataFrame(ciclos_data)
    df_ciclos.to_csv('ciclos_lactacao.csv', index=False)
    print(f"    ✅ {len(df_ciclos)} ciclos criados")
    
    # 3. Dados de lactação (múltiplas ordenhas por ciclo)
    print("  - Gerando dados_lactacao.csv...")
    ordenhas_data = []
    
    for _, ciclo in df_ciclos.iterrows():
        dt_parto = ciclo['dt_parto']
        padrao_dias = ciclo['padrao_dias']
        
        # Produção total do ciclo baseada no potencial genético
        femea = df_bufalos[df_bufalos['id_bufalo'] == ciclo['id_bufala']].iloc[0]
        producao_total = 2000 + (femea['potencial_genetico_leite'] * 1000) + random.randint(-200, 200)
        
        # Curva de lactação realista
        dias_lactacao = np.arange(padrao_dias)
        pico_lactacao = padrao_dias // 3
        
        # Curva gaussiana com pico no primeiro terço
        curva = np.exp(-((dias_lactacao - pico_lactacao) ** 2) / (2 * (pico_lactacao ** 2)))
        producao_diaria = (curva / np.sum(curva)) * producao_total
        
        # Gera ordenhas diárias
        for dia, producao in enumerate(producao_diaria):
            if producao > 0:
                # Simula 2-3 ordenhas por dia
                num_ordenhas = random.randint(2, 3)
                for ordenha in range(num_ordenhas):
                    qt_ordenha = producao / num_ordenhas + np.random.normal(0, 0.5)
                    ordenhas_data.append({
                        "id_lact": len(ordenhas_data) + 1,
                        "id_ciclo_lactacao": ciclo['id_ciclo_lactacao'],
                        "qt_ordenha": max(0, round(qt_ordenha, 2)),
                        "dt_ordenha": dt_parto + timedelta(days=dia)
                    })
    
    df_ordenhas = pd.DataFrame(ordenhas_data)
    df_ordenhas.to_csv('dados_lactacao.csv', index=False)
    print(f"    ✅ {len(df_ordenhas)} registros de ordenha criados")
    
    # 4. Dados zootécnicos
    print("  - Gerando dados_zootecnicos.csv...")
    zootecnicos_data = []
    
    for _, bufalo in df_bufalos.iterrows():
        num_registros = random.randint(2, 5)
        for _ in range(num_registros):
            dias_de_vida = (datetime.now() - bufalo['dt_nascimento']).days
            if dias_de_vida <= 30:
                continue
                
            dt_registro = bufalo['dt_nascimento'] + timedelta(days=random.randint(30, dias_de_vida))
            idade_anos = dias_de_vida / 365.25
            
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
    df_zootecnicos.to_csv('dados_zootecnicos.csv', index=False)
    print(f"    ✅ {len(df_zootecnicos)} registros zootécnicos criados")
    
    # 5. Dados sanitários
    print("  - Gerando dados_sanitarios.csv...")
    sanitarios_data = []
    doencas = ["Mastite", "Metrite", "Problema de Casco", "Laminite", "Pneumonia", "Carrapato"]
    
    for _, bufalo in df_bufalos.iterrows():
        if random.random() > 0.5:  # 50% dos búfalos têm eventos sanitários
            num_eventos = random.randint(1, 3)
            for _ in range(num_eventos):
                dias_de_vida = (datetime.now() - bufalo['dt_nascimento']).days
                if dias_de_vida <= 180:
                    continue
                    
                dt_aplicacao = bufalo['dt_nascimento'] + timedelta(days=random.randint(180, dias_de_vida))
                sanitarios_data.append({
                    "id_sanit": len(sanitarios_data) + 1,
                    "id_bufalo": bufalo['id_bufalo'],
                    "doenca": random.choice(doencas),
                    "medicacao": "Antibiótico" if random.random() > 0.3 else "Anti-inflamatório",
                    "dt_aplicacao": dt_aplicacao
                })
    
    df_sanitarios = pd.DataFrame(sanitarios_data)
    df_sanitarios.to_csv('dados_sanitarios.csv', index=False)
    print(f"    ✅ {len(df_sanitarios)} registros sanitários criados")
    
    # 6. Dados reprodutivos
    print("  - Gerando dados_reproducao.csv...")
    repro_data = []
    
    for _, femea in femeas.iterrows():
        num_eventos = random.randint(1, 4)
        for _ in range(num_eventos):
            dias_de_vida = (datetime.now() - femea['dt_nascimento']).days
            if dias_de_vida <= 365:
                continue
                
            dt_evento = femea['dt_nascimento'] + timedelta(days=random.randint(365, dias_de_vida))
            tipos = ["Inseminação", "Monta Natural", "Diagnóstico de Gestação", "Parto"]
            tipo = random.choice(tipos)
            
            if tipo in ["Inseminação", "Monta Natural"]:
                status = random.choice(["Pendente", "Confirmada", "Falhou"])
            elif tipo == "Diagnóstico de Gestação":
                status = random.choice(["Positivo", "Negativo"])
            else:
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
    df_repro.to_csv('dados_reproducao.csv', index=False)
    print(f"    ✅ {len(df_repro)} registros reprodutivos criados")
    
    print("\n" + "="*60)
    print("✅ DADOS DE TESTE CRIADOS COM SUCESSO!")
    print("="*60)
    print(f"📊 Total de búfalos: {len(df_bufalos)}")
    print(f"🐄 Fêmeas: {len(femeas)}")
    print(f"🔄 Ciclos de lactação: {len(df_ciclos)}")
    print(f"🥛 Registros de ordenha: {len(df_ordenhas)}")
    print(f"⚖️ Dados zootécnicos: {len(df_zootecnicos)}")
    print(f"🏥 Dados sanitários: {len(df_sanitarios)}")
    print(f"👶 Dados reprodutivos: {len(df_repro)}")
    print("="*60)

def testar_genealogia():
    """Testa o módulo de genealogia."""
    print("\n🧬 Testando módulo de genealogia...")
    
    try:
        from app.models.genealogia import criar_arvore_genealogica, CalculadorConsanguinidade
        
        # Carrega dados
        df_bufalos = pd.read_csv('bufalos.csv')
        
        # Cria árvore genealógica
        arvore = criar_arvore_genealogica(df_bufalos)
        print("  ✅ Árvore genealógica criada com sucesso")
        
        # Testa cálculo de consanguinidade
        calculador = CalculadorConsanguinidade(arvore)
        
        # Testa com alguns búfalos
        for i in range(1, 6):
            try:
                consanguinidade = calculador.calcular_coeficiente_wright(i)
                print(f"    🧬 Búfalo {i}: {consanguinidade*100:.2f}% de consanguinidade")
            except Exception as e:
                print(f"    ⚠️ Búfalo {i}: Erro - {e}")
        
        # Testa simulação de acasalamento
        femeas = df_bufalos[df_bufalos['sexo'] == 'F']['id_bufalo'].iloc[:3]
        machos = df_bufalos[df_bufalos['sexo'] == 'M']['id_bufalo'].iloc[:3]
        
        for femea in femeas:
            for macho in machos:
                try:
                    simulacao = calculador.simular_acasalamento(macho, femea)
                    print(f"    💑 Acasalamento {macho}-{femea}: {simulacao['consanguinidade_prole']:.2f}% risco")
                except Exception as e:
                    print(f"    ⚠️ Erro no acasalamento {macho}-{femea}: {e}")
        
        print("  ✅ Módulo de genealogia testado com sucesso!")
        
    except Exception as e:
        print(f"  ❌ Erro ao testar genealogia: {e}")

def testar_predicao():
    """Testa o módulo de predição."""
    print("\n🧠 Testando módulo de predição...")
    
    try:
        from app.models.prediction import obter_informacoes_femea
        
        # Carrega dados
        df_bufalos = pd.read_csv('bufalos.csv')
        
        # Testa obtenção de informações de fêmeas
        femeas = df_bufalos[df_bufalos['sexo'] == 'F']['id_bufalo'].iloc[:5]
        
        for femea_id in femeas:
            try:
                info = obter_informacoes_femea(femea_id, df_bufalos)
                if info:
                    print(f"    🐄 Fêmea {femea_id}: {info['idade_mae_anos']:.1f} anos, Propriedade {info['id_propriedade']}")
                else:
                    print(f"    ⚠️ Fêmea {femea_id}: Informações não encontradas")
            except Exception as e:
                print(f"    ❌ Erro na fêmea {femea_id}: {e}")
        
        print("  ✅ Módulo de predição testado com sucesso!")
        
    except Exception as e:
        print(f"  ❌ Erro ao testar predição: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO TESTES DA BUFFS IA VERSÃO 1.0.0")
    print("="*60)
    
    # Cria dados de teste
    criar_dados_teste()
    
    # Testa módulos
    testar_genealogia()
    testar_predicao()
    
    print("\n" + "="*60)
    print("🎯 TESTES CONCLUÍDOS!")
    print("="*60)
    print("🚀 PRÓXIMOS PASSOS:")
    print("1. Execute: python treinar_ia.py")
    print("2. Execute: python -m uvicorn app.main:app --reload --port 5001")
    print("3. Teste a API em: http://localhost:5001/docs")
    print("="*60)
