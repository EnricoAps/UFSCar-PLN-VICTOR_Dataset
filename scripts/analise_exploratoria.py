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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.preprocessamento import CLASSES, ROTULO_NAO_ROTULADO

sns.set_theme(style="whitegrid")


def _nome_classe(c):
    """Retorna um rotulo legivel para a classe (inclui o nao-rotulado)."""
    if c == ROTULO_NAO_ROTULADO:
        return "Nao rotulado (-1)"
    return f"{c} - {CLASSES.get(c, '?')}"


def resumo_geral(df, coluna_classe="Category"):
    """Imprime um resumo descritivo da base."""
    print(f"Numero de amostras : {len(df)}")
    print(f"Numero de colunas  : {df.shape[1]}")
    print(f"Colunas            : {list(df.columns)}")
    print(f"Valores ausentes   :\n{df.isna().sum()}\n")
    if coluna_classe in df.columns:
        print("Distribuicao de classes:")
        dist = df[coluna_classe].value_counts().sort_index()
        for c, n in dist.items():
            print(f"  {_nome_classe(c):<22} : {n:6d} ({100 * n / len(df):5.2f}%)")


def plotar_distribuicao_classes(df, coluna_classe="Category", ax=None):
    """Grafico de barras com a distribuicao das classes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    dist = df[coluna_classe].value_counts().sort_index()
    rotulos = [_nome_classe(c) for c in dist.index]
    sns.barplot(x=rotulos, y=dist.values, ax=ax, palette="viridis")
    ax.set_title("Distribuicao das classes")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Frequencia")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(dist.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    return ax


def estatisticas_tamanho(df, coluna_texto="Body"):
    """Calcula o numero de caracteres e de tokens por documento."""
    df = df.copy()
    df["n_caracteres"] = df[coluna_texto].str.len()
    df["n_tokens"] = df[coluna_texto].str.split().map(len)
    print(df[["n_caracteres", "n_tokens"]].describe())
    return df


def plotar_tamanho_por_classe(df, coluna_classe="Category",
                              coluna_tokens="n_tokens", ax=None):
    """Boxplot do tamanho (tokens) dos documentos por classe."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    df = df.copy()
    df["classe"] = df[coluna_classe].map(_nome_classe)
    sns.boxplot(data=df, x="classe", y=coluna_tokens, ax=ax, showfliers=False,
                palette="viridis")
    ax.set_title("Numero de tokens por classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Tokens")
    ax.tick_params(axis="x", rotation=30)
    return ax


def palavras_frequentes(df, coluna_texto="texto_limpo", n=20):
    """Retorna as n palavras mais frequentes no corpus inteiro."""
    contador = Counter()
    for texto in df[coluna_texto]:
        contador.update(str(texto).split())
    return contador.most_common(n)


def palavras_frequentes_por_classe(df, coluna_texto="texto_limpo",
                                   coluna_classe="Category", n=15):
    """Imprime as n palavras mais frequentes de cada classe."""
    for c in sorted(df[coluna_classe].unique()):
        sub = df[df[coluna_classe] == c]
        contador = Counter()
        for texto in sub[coluna_texto]:
            contador.update(str(texto).split())
        top = ", ".join(p for p, _ in contador.most_common(n))
        print(f"[{_nome_classe(c)}]\n  {top}\n")


def plotar_nuvem_frequencia(df, coluna_texto="texto_limpo", n=20, ax=None):
    """Barra horizontal das palavras mais frequentes (alternativa a wordcloud)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    pares = palavras_frequentes(df, coluna_texto, n)
    palavras = [p for p, _ in pares][::-1]
    freqs = [f for _, f in pares][::-1]
    sns.barplot(x=freqs, y=palavras, ax=ax, palette="mako")
    ax.set_title(f"{n} palavras mais frequentes")
    ax.set_xlabel("Frequencia")
    return ax
