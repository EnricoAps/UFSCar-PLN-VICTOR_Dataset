# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
#
# Nome: Thales Leonardo Euler Vieira de Sousa   RA: 822881
# Nome: Enrico Augusto Pagani da Silva          RA: 822888
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import re
import unicodedata

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Garante que os recursos do NLTK necessarios estejam disponiveis.
for _recurso, _caminho in [
    ("stopwords", "corpora/stopwords"),
    ("rslp", "stemmers/rslp"),
]:
    try:
        nltk.data.find(_caminho)
    except LookupError:
        nltk.download(_recurso, quiet=True)


# Rotulo usado na base VICTOR
ROTULO_NAO_ROTULADO = -1

# Mapa de classes do problema (apenas as 5 categorias validas).
CLASSES = {
    0: "Acordao",
    1: "ARE",
    2: "Despacho",
    3: "RE",
    4: "Sentenca",
}

def corrigir_encoding(texto):
    """Corrige o "mojibake" da base (UTF-8 lido como Latin-1).

    Ex.: "conclusAo" mal-codificado vira "conclusao" corretamente acentuado.
    A correcao e aplicada de forma defensiva: se a string ja estiver correta,
    o texto original e retornado sem alteracao.
    """
    if not isinstance(texto, str):
        return ""
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def carregar_dados(caminho, corrigir=True):
    """Le um CSV da base VICTOR e (opcionalmente) corrige o encoding do texto.

    Parametros
    ----------
    caminho : str
        Caminho para o arquivo (train.csv / test.csv).
    corrigir : bool
        Se True, aplica a correcao de encoding na coluna 'Body'.
    """
    df = pd.read_csv(caminho, encoding="utf-8")
    if corrigir and "Body" in df.columns:
        df["Body"] = df["Body"].map(corrigir_encoding)
    return df


# O texto vem embrulhado em um literal do tipo {"...."}.
_PADRAO_WRAPPER = re.compile(r'^\s*\{\s*"|"\s*\}\s*$')

# Tokens artificiais da base (ex.: ARTIGO_102). O numero do artigo e mantido
# como parte do token (artigo_102), pois e altamente discriminativo da classe.
_PADRAO_ARTIGO = re.compile(r"\bARTIGO_(\d+)\b", flags=re.IGNORECASE)

# Mantendo numeros: remove apenas pontuacao/ruido (preserva letras, digitos e "_").
_PADRAO_NAO_ALFANUM = re.compile(r"[^a-z0-9_\s]")
# Removendo numeros: mantem apenas letras.
_PADRAO_NAO_LETRAS = re.compile(r"[^a-z\s]")
_PADRAO_ESPACOS = re.compile(r"\s+")


def remover_acentos(texto):
    """Remove acentos preservando a letra base (NFKD)."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar_texto(texto, remover_acento=True, manter_numeros=True):
    """Limpeza basica de um documento juridico.

    Etapas: desembrulha {"..."}, preserva marcadores ARTIGO_xxx (como artigo_xxx),
    minusculiza, remove pontuacao/ruido de OCR e normaliza espacos.

    `manter_numeros=True` preserva digitos e o token do artigo (ex.: artigo_102),
    que sao fortemente discriminativos do tipo de peca (ganho de ~+0.007 F1).
    """
    if not isinstance(texto, str):
        return ""

    texto = _PADRAO_WRAPPER.sub("", texto)
    texto = _PADRAO_ARTIGO.sub(r" artigo_\1 ", texto)
    texto = texto.lower()
    if remover_acento:
        texto = remover_acentos(texto)
    if manter_numeros:
        texto = _PADRAO_NAO_ALFANUM.sub(" ", texto)
    else:
        texto = _PADRAO_NAO_LETRAS.sub(" ", texto)
    texto = _PADRAO_ESPACOS.sub(" ", texto).strip()
    return texto

_STOPWORDS_PT = set(remover_acentos(p) for p in stopwords.words("portuguese"))
_STEMMER = RSLPStemmer()


def remover_stopwords(texto, extra=None):
    """Remove stopwords do portugues (acentos ja removidos para casar)."""
    stop = _STOPWORDS_PT if extra is None else _STOPWORDS_PT | set(extra)
    return " ".join(t for t in texto.split() if t not in stop and len(t) > 2)


def aplicar_stemming(texto):
    """Aplica o stemmer RSLP (portugues) token a token."""
    return " ".join(_STEMMER.stem(t) for t in texto.split())


def preprocessar_texto(texto, usar_stopwords=True, usar_stemming=False,
                       manter_numeros=True):
    """Pipeline completo de pre-processamento de um unico documento.

    Configuracao padrao (manter_numeros=True, sem stemming) foi a melhor em
    validacao cruzada: F1-weighted 0.9298 vs 0.9231 da limpeza agressiva.
    """
    texto = limpar_texto(texto, manter_numeros=manter_numeros)
    if usar_stopwords:
        texto = remover_stopwords(texto)
    if usar_stemming:
        texto = aplicar_stemming(texto)
    return texto


def preprocessar_dataframe(df, coluna="Body", usar_stopwords=True,
                           usar_stemming=False, manter_numeros=True,
                           nova_coluna="texto_limpo"):
    """Aplica o pre-processamento a uma coluna inteira do DataFrame."""
    df = df.copy()
    df[nova_coluna] = df[coluna].map(
        lambda t: preprocessar_texto(t, usar_stopwords, usar_stemming,
                                     manter_numeros)
    )
    return df


def filtrar_rotulados(df, coluna_classe="Category"):
    """Remove as amostras NAO rotuladas (Category == -1) do treino."""
    return df[df[coluna_classe] != ROTULO_NAO_ROTULADO].reset_index(drop=True)
