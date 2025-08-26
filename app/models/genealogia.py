# =================================================================
# ARQUIVO: genealogia.py (VERSÃO 1.0.0)
# OBJETIVO: Módulo para análise genealógica e cálculo de consanguinidade
#           em búfalos usando o coeficiente de Wright.
# =================================================================
import pandas as pd
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

class ArvoreGenealogica:
    """
    Classe para construção e análise de árvores genealógicas de búfalos.
    """
    
    def __init__(self, df_bufalos: pd.DataFrame):
        """
        Inicializa a árvore genealógica a partir dos dados dos búfalos.
        
        Args:
            df_bufalos: DataFrame com colunas id_bufalo, id_pai, id_mae
        """
        self.df_bufalos = df_bufalos
        self.arvore = {}
        self._construir_arvore()
    
    def _construir_arvore(self):
        """Constrói a representação da árvore genealógica."""
        for _, bufalo in self.df_bufalos.iterrows():
            id_bufalo = bufalo['id_bufalo']
            id_pai = bufalo.get('id_pai')
            id_mae = bufalo.get('id_mae')
            
            self.arvore[id_bufalo] = {
                'pai': id_pai,
                'mae': id_mae,
                'filhos': []
            }
            
            # Adiciona como filho dos pais
            if id_pai and pd.notna(id_pai):
                if id_pai in self.arvore:
                    self.arvore[id_pai]['filhos'].append(id_bufalo)
                else:
                    self.arvore[id_pai] = {'pai': None, 'mae': None, 'filhos': [id_bufalo]}
            
            if id_mae and pd.notna(id_mae):
                if id_mae in self.arvore:
                    self.arvore[id_mae]['filhos'].append(id_bufalo)
                else:
                    self.arvore[id_mae] = {'pai': None, 'mae': None, 'filhos': [id_bufalo]}
    
    def obter_ancestrais(self, id_bufalo: int, max_geracoes: int = 5) -> Dict[str, List[int]]:
        """
        Obtém ancestrais de um búfalo até uma geração específica.
        
        Args:
            id_bufalo: ID do búfalo
            max_geracoes: Número máximo de gerações para buscar
            
        Returns:
            Dicionário com ancestrais por geração
        """
        if id_bufalo not in self.arvore:
            return {}
        
        ancestrais = defaultdict(list)
        visitados = set()
        fila = deque([(id_bufalo, 0)])
        
        while fila:
            bufalo_atual, geracao = fila.popleft()
            
            if geracao >= max_geracoes or bufalo_atual in visitados:
                continue
            
            visitados.add(bufalo_atual)
            bufalo_info = self.arvore.get(bufalo_atual, {})
            
            pai = bufalo_info.get('pai')
            mae = bufalo_info.get('mae')
            
            if pai and pd.notna(pai):
                ancestrais[f'geracao_{geracao + 1}_pai'].append(pai)
                fila.append((pai, geracao + 1))
            
            if mae and pd.notna(mae):
                ancestrais[f'geracao_{geracao + 1}_mae'].append(mae)
                fila.append((mae, geracao + 1))
        
        return dict(ancestrais)
    
    def obter_descendentes(self, id_bufalo: int, max_geracoes: int = 3) -> Dict[str, List[int]]:
        """
        Obtém descendentes de um búfalo até uma geração específica.
        
        Args:
            id_bufalo: ID do búfalo
            max_geracoes: Número máximo de gerações para buscar
            
        Returns:
            Dicionário com descendentes por geração
        """
        if id_bufalo not in self.arvore:
            return {}
        
        descendentes = defaultdict(list)
        visitados = set()
        fila = deque([(id_bufalo, 0)])
        
        while fila:
            bufalo_atual, geracao = fila.popleft()
            
            if geracao >= max_geracoes or bufalo_atual in visitados:
                continue
            
            visitados.add(bufalo_atual)
            bufalo_info = self.arvore.get(bufalo_atual, {})
            
            for filho in bufalo_info.get('filhos', []):
                descendentes[f'geracao_{geracao + 1}_filhos'].append(filho)
                fila.append((filho, geracao + 1))
        
        return dict(descendentes)
    
    def encontrar_ancestrais_comuns(self, id_bufalo1: int, id_bufalo2: int) -> Set[int]:
        """
        Encontra ancestrais comuns entre dois búfalos.
        
        Args:
            id_bufalo1: ID do primeiro búfalo
            id_bufalo2: ID do segundo búfalo
            
        Returns:
            Conjunto de IDs dos ancestrais comuns
        """
        ancestrais1 = set()
        ancestrais2 = set()
        
        # Coleta ancestrais do primeiro búfalo
        fila1 = deque([(id_bufalo1, 0)])
        while fila1:
            bufalo, geracao = fila1.popleft()
            if geracao > 10:  # Limite de segurança
                continue
            
            bufalo_info = self.arvore.get(bufalo, {})
            pai = bufalo_info.get('pai')
            mae = bufalo_info.get('mae')
            
            if pai and pd.notna(pai):
                ancestrais1.add(pai)
                fila1.append((pai, geracao + 1))
            
            if mae and pd.notna(mae):
                ancestrais1.add(mae)
                fila1.append((mae, geracao + 1))
        
        # Coleta ancestrais do segundo búfalo
        fila2 = deque([(id_bufalo2, 0)])
        while fila2:
            bufalo, geracao = fila2.popleft()
            if geracao > 10:  # Limite de segurança
                continue
            
            bufalo_info = self.arvore.get(bufalo, {})
            pai = bufalo_info.get('pai')
            mae = bufalo_info.get('mae')
            
            if pai and pd.notna(pai):
                ancestrais2.add(pai)
                fila2.append((pai, geracao + 1))
            
            if mae and pd.notna(mae):
                ancestrais2.add(mae)
                fila2.append((mae, geracao + 1))
        
        return ancestrais1.intersection(ancestrais2)

class CalculadorConsanguinidade:
    """
    Classe para cálculo de coeficientes de consanguinidade e parentesco.
    """
    
    def __init__(self, arvore: ArvoreGenealogica):
        """
        Inicializa o calculador com uma árvore genealógica.
        
        Args:
            arvore: Instância de ArvoreGenealogica
        """
        self.arvore = arvore
        self.cache_consanguinidade = {}
        self.cache_parentesco = {}
    
    def calcular_coeficiente_wright(self, id_bufalo: int) -> float:
        """
        Calcula o coeficiente de consanguinidade de Wright para um búfalo.
        
        Args:
            id_bufalo: ID do búfalo
            
        Returns:
            Coeficiente de consanguinidade (0.0 a 1.0)
        """
        if id_bufalo in self.cache_consanguinidade:
            return self.cache_consanguinidade[id_bufalo]
        
        if id_bufalo not in self.arvore.arvore:
            return 0.0
        
        bufalo_info = self.arvore.arvore[id_bufalo]
        pai = bufalo_info.get('pai')
        mae = bufalo_info.get('mae')
        
        # Se não tem pais, não há consanguinidade
        if not pai or not mae or pd.isna(pai) or pd.isna(mae):
            self.cache_consanguinidade[id_bufalo] = 0.0
            return 0.0
        
        # Calcula consanguinidade dos pais
        consanguinidade_pai = self.calcular_coeficiente_wright(pai)
        consanguinidade_mae = self.calcular_coeficiente_wright(mae)
        
        # Calcula coeficiente de parentesco entre os pais
        parentesco_pais = self.calcular_coeficiente_parentesco(pai, mae)
        
        # Fórmula de Wright: F = (1 + F_pai + F_mae) * r_pai_mae / 2
        consanguinidade = (1 + consanguinidade_pai + consanguinidade_mae) * parentesco_pais / 2
        
        self.cache_consanguinidade[id_bufalo] = consanguinidade
        return consanguinidade
    
    def calcular_coeficiente_parentesco(self, id_bufalo1: int, id_bufalo2: int) -> float:
        """
        Calcula o coeficiente de parentesco entre dois búfalos.
        
        Args:
            id_bufalo1: ID do primeiro búfalo
            id_bufalo2: ID do segundo búfalo
            
        Returns:
            Coeficiente de parentesco (0.0 a 1.0)
        """
        cache_key = tuple(sorted([id_bufalo1, id_bufalo2]))
        if cache_key in self.cache_parentesco:
            return self.cache_parentesco[cache_key]
        
        if id_bufalo1 == id_bufalo2:
            return 1.0
        
        # Encontra ancestrais comuns
        ancestrais_comuns = self.arvore.encontrar_ancestrais_comuns(id_bufalo1, id_bufalo2)
        
        if not ancestrais_comuns:
            self.cache_parentesco[cache_key] = 0.0
            return 0.0
        
        # Calcula parentesco usando o método dos caminhos
        parentesco_total = 0.0
        
        for ancestral in ancestrais_comuns:
            caminhos1 = self._encontrar_caminhos_para_ancestral(id_bufalo1, ancestral)
            caminhos2 = self._encontrar_caminhos_para_ancestral(id_bufalo2, ancestral)
            
            for caminho1 in caminhos1:
                for caminho2 in caminhos2:
                    # Calcula contribuição deste ancestral
                    contribuicao = (0.5) ** (len(caminho1) + len(caminho2) - 2)
                    
                    # Aplica consanguinidade do ancestral
                    consanguinidade_ancestral = self.calcular_coeficiente_wright(ancestral)
                    contribuicao *= (1 + consanguinidade_ancestral)
                    
                    parentesco_total += contribuicao
        
        self.cache_parentesco[cache_key] = parentesco_total
        return parentesco_total
    
    def _encontrar_caminhos_para_ancestral(self, id_bufalo: int, id_ancestral: int) -> List[List[int]]:
        """
        Encontra todos os caminhos de um búfalo até um ancestral.
        
        Args:
            id_bufalo: ID do búfalo
            id_ancestral: ID do ancestral
            
        Returns:
            Lista de caminhos (cada caminho é uma lista de IDs)
        """
        if id_bufalo == id_ancestral:
            return [[]]
        
        if id_bufalo not in self.arvore.arvore:
            return []
        
        caminhos = []
        bufalo_info = self.arvore.arvore[id_bufalo]
        pai = bufalo_info.get('pai')
        mae = bufalo_info.get('mae')
        
        # Busca pelo pai
        if pai and pd.notna(pai):
            caminhos_pai = self._encontrar_caminhos_para_ancestral(pai, id_ancestral)
            for caminho in caminhos_pai:
                caminhos.append([pai] + caminho)
        
        # Busca pela mãe
        if mae and pd.notna(mae):
            caminhos_mae = self._encontrar_caminhos_para_ancestral(mae, id_ancestral)
            for caminho in caminhos_mae:
                caminhos.append([mae] + caminho)
        
        return caminhos
    
    def simular_acasalamento(self, id_macho: int, id_femea: int) -> Dict[str, Any]:
        """
        Simula um acasalamento e calcula a consanguinidade da prole.
        
        Args:
            id_macho: ID do macho
            id_femea: ID da fêmea
            
        Returns:
            Dicionário com resultados da simulação
        """
        # Calcula consanguinidade dos pais
        consanguinidade_macho = self.calcular_coeficiente_wright(id_macho)
        consanguinidade_femea = self.calcular_coeficiente_wright(id_femea)
        
        # Calcula parentesco entre os pais
        parentesco_pais = self.calcular_coeficiente_parentesco(id_macho, id_femea)
        
        # Calcula consanguinidade da prole
        consanguinidade_prole = parentesco_pais / 2
        
        # Classifica risco
        if consanguinidade_prole > 0.0625:  # > 6.25%
            risco = "Alto"
            recomendacao = "Evitar acasalamento - risco genético elevado"
        elif consanguinidade_prole > 0.03125:  # > 3.125%
            risco = "Médio"
            recomendacao = "Acasalamento com cautela - monitorar prole"
        else:
            risco = "Baixo"
            recomendacao = "Acasalamento seguro - baixo risco genético"
        
        return {
            "macho_id": id_macho,
            "femea_id": id_femea,
            "consanguinidade_macho": round(consanguinidade_macho * 100, 2),
            "consanguinidade_femea": round(consanguinidade_femea * 100, 2),
            "parentesco_pais": round(parentesco_pais * 100, 2),
            "consanguinidade_prole": round(consanguinidade_prole * 100, 2),
            "risco_consanguinidade": risco,
            "recomendacao": recomendacao
        }
    
    def encontrar_machos_compatíveis(
        self, 
        id_femea: int, 
        max_consanguinidade: float = 0.0625
    ) -> List[Dict[str, Any]]:
        """
        Encontra machos compatíveis para uma fêmea baseado na consanguinidade.
        
        Args:
            id_femea: ID da fêmea
            max_consanguinidade: Consanguinidade máxima aceitável (padrão: 6.25%)
            
        Returns:
            Lista de machos compatíveis ordenados por compatibilidade
        """
        # Filtra apenas machos
        machos = self.arvore.df_bufalos[self.arvore.df_bufalos['sexo'] == 'M']
        
        machos_compatíveis = []
        
        for _, macho in machos.iterrows():
            id_macho = macho['id_bufalo']
            
            # Simula acasalamento
            simulacao = self.simular_acasalamento(id_macho, id_femea)
            
            # Verifica se está dentro do limite
            if simulacao['consanguinidade_prole'] <= max_consanguinidade * 100:
                machos_compatíveis.append({
                    "id_macho": id_macho,
                    "consanguinidade_prole": simulacao['consanguinidade_prole'],
                    "parentesco_pais": simulacao['parentesco_pais'],
                    "risco": simulacao['risco_consanguinidade'],
                    "recomendacao": simulacao['recomendacao']
                })
        
        # Ordena por consanguinidade (menor primeiro)
        machos_compatíveis.sort(key=lambda x: x['consanguinidade_prole'])
        
        return machos_compatíveis

def criar_arvore_genealogica(df_bufalos: pd.DataFrame) -> ArvoreGenealogica:
    """
    Função auxiliar para criar uma árvore genealógica.
    
    Args:
        df_bufalos: DataFrame com dados dos búfalos
        
    Returns:
        Instância de ArvoreGenealogica
    """
    return ArvoreGenealogica(df_bufalos)
