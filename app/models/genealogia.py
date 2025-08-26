# =================================================================
# ARQUIVO: genealogia.py (VERSÃO 1.1 - WRIGHT CORRIGIDO E ROBUSTO)
# OBJETIVO: Implementação robusta e correta para cálculo de
#           consanguinidade e parentesco usando o Coeficiente de Wright.
# =================================================================
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class CalculadorConsanguinidade:
    """
    Calcula o parentesco e a consanguinidade usando o método recursivo
    de Wright com cache (memoização) para garantir performance e evitar erros.
    """
    def __init__(self, df_bufalos: pd.DataFrame):
        self._df_bufalos = df_bufalos.copy()
        
        # Caches para armazenar resultados já calculados e evitar reprocessamento
        self._consanguinidade_cache: Dict[int, float] = {}
        self._parentesco_cache: Dict[Tuple[int, int], float] = {}
        
        # Lida com DataFrame vazio de forma segura para evitar erros na inicialização
        if self._df_bufalos.empty:
            self._pedigree = {}
            return

        # Prepara o pedigree para os cálculos
        self._df_bufalos['id_pai'] = self._df_bufalos['id_pai'].fillna(0).astype(int)
        self._df_bufalos['id_mae'] = self._df_bufalos['id_mae'].fillna(0).astype(int)
        self._pedigree = self._df_bufalos.set_index('id_bufalo')[['id_pai', 'id_mae']].to_dict('index')

    def _get_pais(self, animal_id: int) -> Tuple[int, int]:
        """Retorna (pai, mae) de um animal. Retorna (0, 0) se não encontrado."""
        if animal_id == 0:
            return 0, 0
        info = self._pedigree.get(animal_id)
        return (info['id_pai'], info['id_mae']) if info else (0, 0)

    def calcular_parentesco(self, id_a: int, id_b: int) -> float:
        """Calcula o coeficiente de parentesco (coancestria) entre dois animais A e B."""
        # Ordena para garantir consistência no cache
        if id_a > id_b:
            id_a, id_b = id_b, id_a
        
        cache_key = (id_a, id_b)
        if cache_key in self._parentesco_cache:
            return self._parentesco_cache[cache_key]

        # Caso base: parentesco de um animal com ele mesmo
        if id_a == id_b:
            resultado = 0.5 * (1 + self.calcular_consanguinidade(id_a))
            self._parentesco_cache[cache_key] = resultado
            return resultado

        # Caso base: se um dos animais é fundador (sem pais), o parentesco é zero
        pai_b, mae_b = self._get_pais(id_b)
        if pai_b == 0 and mae_b == 0:
            self._parentesco_cache[cache_key] = 0.0
            return 0.0
        
        # Fórmula recursiva de Wright: o parentesco de A e B é a média do parentesco de A com os pais de B
        resultado = 0.5 * (self.calcular_parentesco(id_a, pai_b) + self.calcular_parentesco(id_a, mae_b))
        self._parentesco_cache[cache_key] = resultado
        return resultado

    def calcular_consanguinidade(self, animal_id: int) -> float:
        """Calcula o coeficiente de consanguinidade de um animal."""
        if animal_id in self._consanguinidade_cache:
            return self._consanguinidade_cache[animal_id]

        pai_id, mae_id = self._get_pais(animal_id)

        if pai_id == 0 or mae_id == 0:
            self._consanguinidade_cache[animal_id] = 0.0
            return 0.0

        # A consanguinidade de um indivíduo é o parentesco (coancestria) entre seus pais
        resultado = self.calcular_parentesco(pai_id, mae_id)
        self._consanguinidade_cache[animal_id] = resultado
        return resultado

    def simular_acasalamento(self, id_macho: int, id_femea: int) -> Dict:
        """Simula um acasalamento e retorna um dicionário com os resultados."""
        if not self._pedigree:
            raise ValueError("Dados de genealogia não foram carregados. Não é possível simular.")

        consanguinidade_macho = self.calcular_consanguinidade(id_macho)
        consanguinidade_femea = self.calcular_consanguinidade(id_femea)
        parentesco_pais = self.calcular_parentesco(id_macho, id_femea)
        
        # A consanguinidade da prole é o parentesco dos pais
        consanguinidade_prole = parentesco_pais

        risco = "Baixo"
        recomendacao = "Acasalamento seguro - baixo risco genético."
        if consanguinidade_prole > 0.125: # Acima de primos-irmãos (ex: pai-filha = 0.25)
            risco = "Extremo"
            recomendacao = "NÃO RECOMENDADO. Risco genético extremo (ex: pai-filha, irmãos completos)."
        elif consanguinidade_prole > 0.0625: # Acima de meio-irmãos
            risco = "Alto"
            recomendacao = "Atenção: Risco genético alto. Evitar se possível."
        
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

# --- Funções Auxiliares para serem chamadas pelo main.py ---
# Mantém a compatibilidade com a sua API
def criar_arvore_genealogica(df_bufalos: pd.DataFrame):
    """Função de compatibilidade. O Calculador lida com o DataFrame diretamente."""
    return df_bufalos
