import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SupabaseConnection:
    def __init__(self):
        self.db_url = os.getenv('SUPABASE_DB_URL')
        self.engine = None
        
        if self.db_url:
            try:
                self.engine = create_engine(self.db_url, pool_pre_ping=True)
                logger.info("✅ Conexão com Supabase configurada")
            except Exception as e:
                logger.error(f"❌ Erro ao conectar com Supabase: {e}")
        else:
            logger.warning("⚠️ SUPABASE_DB_URL não configurada, usando dados sintéticos")
    
    def test_connection(self) -> bool:
        """Testa a conexão com o banco."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"❌ Erro na conexão: {e}")
            return False
    
    def get_bufalos_data(self) -> pd.DataFrame:
        """Busca dados completos dos búfalos."""
        if not self.engine:
            logger.info("📁 Usando dados sintéticos (CSV)")
            return pd.read_csv('bufalos.csv', parse_dates=['dt_nascimento'])
        
        try:
            query = """
            SELECT 
                b.id_bufalo,
                b.nome,
                b.sexo,
                b.dt_nascimento,
                b.id_pai,
                b.id_mae,
                b.id_raca,
                b.id_propriedade,
                b.status,
                1.0 as potencial_genetico_leite  -- Campo calculado/padrão
            FROM "Bufalo" b
            WHERE b.status = true  -- Apenas búfalos ativos
            ORDER BY b.id_bufalo
            """
            
            df = pd.read_sql(query, self.engine, parse_dates=['dt_nascimento'])
            logger.info(f"✅ Carregados {len(df)} búfalos do Supabase")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar búfalos: {e}")
            logger.info("📁 Usando dados sintéticos como fallback")
            return pd.read_csv('bufalos.csv', parse_dates=['dt_nascimento'])
    
    def get_ciclos_lactacao(self) -> pd.DataFrame:
        """Busca dados dos ciclos de lactação."""
        if not self.engine:
            return pd.read_csv('ciclos_lactacao.csv', parse_dates=['dt_parto', 'dt_secagem_real'])
        
        try:
            query = """
            SELECT 
                cl.id_ciclo_lactacao,
                cl.id_bufala,
                cl.dt_parto,
                cl.dt_secagem_real,
                cl.padrao_dias,
                cl.status
            FROM "CicloLactacao" cl
            ORDER BY cl.dt_parto DESC
            """
            
            df = pd.read_sql(query, self.engine, parse_dates=['dt_parto', 'dt_secagem_real'])
            logger.info(f"✅ Carregados {len(df)} ciclos de lactação")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar ciclos: {e}")
            return pd.read_csv('ciclos_lactacao.csv', parse_dates=['dt_parto', 'dt_secagem_real'])
    
    def get_dados_lactacao(self) -> pd.DataFrame:
        """Busca dados de produção de leite."""
        if not self.engine:
            return pd.read_csv('dados_lactacao.csv')
        
        try:
            query = """
            SELECT 
                dl.id_lact,
                dl.id_bufala,
                dl.id_ciclo_lactacao,
                dl.qt_ordenha,
                dl.dt_ordenha,
                dl.periodo
            FROM "DadosLactacao" dl
            ORDER BY dl.dt_ordenha DESC
            """
            
            df = pd.read_sql(query, self.engine, parse_dates=['dt_ordenha'])
            logger.info(f"✅ Carregados {len(df)} registros de lactação")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados de lactação: {e}")
            return pd.read_csv('dados_lactacao.csv')
    
    def get_dados_zootecnicos(self) -> pd.DataFrame:
        """Busca dados zootécnicos."""
        if not self.engine:
            try:
                return pd.read_csv('dados_zootecnicos.csv', parse_dates=['dt_registro'])
            except FileNotFoundError:
                return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                dz.id_zootec,
                dz.id_bufalo,
                dz.peso,
                dz.condicao_corporal,
                dz.dt_registro,
                dz.tipo_pesagem
            FROM "DadosZootecnicos" dz
            ORDER BY dz.dt_registro DESC
            """
            
            df = pd.read_sql(query, self.engine, parse_dates=['dt_registro'])
            logger.info(f"✅ Carregados {len(df)} registros zootécnicos")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados zootécnicos: {e}")
            return pd.DataFrame()
    
    def get_dados_sanitarios(self) -> pd.DataFrame:
        """Busca dados sanitários."""
        if not self.engine:
            try:
                return pd.read_csv('dados_sanitarios.csv', parse_dates=['dt_aplicacao'])
            except FileNotFoundError:
                return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                ds.id_sanit,
                ds.id_bufalo,
                ds.tipo_tratamento,
                ds.medicacao,
                ds.dt_aplicacao,
                ds.doenca
            FROM "DadosSanitarios" ds
            ORDER BY ds.dt_aplicacao DESC
            """
            
            df = pd.read_sql(query, self.engine, parse_dates=['dt_aplicacao'])
            logger.info(f"✅ Carregados {len(df)} registros sanitários")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados sanitários: {e}")
            return pd.DataFrame()
    
    def get_dados_reproducao(self) -> pd.DataFrame:
        """Busca dados reprodutivos."""
        if not self.engine:
            try:
                return pd.read_csv('dados_reproducao.csv', parse_dates=['dt_evento'])
            except FileNotFoundError:
                return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                dr.id_reproducao,
                dr.id_receptora,
                dr.tipo_inseminacao,
                dr.status,
                dr.dt_evento
            FROM "DadosReproducao" dr
            ORDER BY dr.dt_evento DESC
            """
            
            df = pd.read_sql(query, self.engine, parse_dates=['dt_evento'])
            logger.info(f"✅ Carregados {len(df)} registros reprodutivos")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar dados reprodutivos: {e}")
            return pd.DataFrame()
    
    def get_bufalo_by_id(self, id_bufalo: int) -> Optional[dict]:
        """Busca um búfalo específico por ID."""
        if not self.engine:
            return None
        
        try:
            query = """
            SELECT 
                b.id_bufalo,
                b.nome,
                b.sexo,
                b.dt_nascimento,
                b.id_pai,
                b.id_mae,
                b.id_raca,
                b.id_propriedade,
                b.status
            FROM "Bufalo" b
            WHERE b.id_bufalo = :id_bufalo AND b.status = true
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {"id_bufalo": id_bufalo})
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao buscar búfalo {id_bufalo}: {e}")
            return None

# Instância global
supabase_db = SupabaseConnection()