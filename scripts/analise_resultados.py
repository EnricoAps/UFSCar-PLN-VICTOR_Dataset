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

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)

from scripts.preprocessamento import CLASSES

sns.set_theme(style="whitegrid")
_NOMES = [CLASSES[c] for c in sorted(CLASSES)]


def relatorio_classificacao(y_true, y_pred):
    """Imprime precision/recall/F1 por classe e o F1 macro/weighted."""
    print(classification_report(y_true, y_pred, target_names=_NOMES, digits=4))
    print(f"F1 macro    : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 weighted : {f1_score(y_true, y_pred, average='weighted'):.4f}")


def plotar_matriz_confusao(y_true, y_pred, normalizar=True, ax=None):
    """Plota a matriz de confusao (opcionalmente normalizada por linha)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred,
                          normalize="true" if normalizar else None)
    disp = ConfusionMatrixDisplay(cm, display_labels=_NOMES)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalizar else "d",
              colorbar=False)
    ax.set_title("Matriz de confusao" + (" (normalizada)" if normalizar else ""))
    ax.tick_params(axis="x", rotation=30)
    return ax


def comparar_f1_por_classe(predicoes, y_true):
    """Tabela de F1 por classe para varios modelos {nome: y_pred}."""
    linhas = []
    for nome, y_pred in predicoes.items():
        f1s = f1_score(y_true, y_pred, average=None)
        linha = {"modelo": nome}
        linha.update({n: round(f, 4) for n, f in zip(_NOMES, f1s)})
        linha["macro"] = round(f1_score(y_true, y_pred, average="macro"), 4)
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("modelo")


def plotar_comparacao_modelos(tabela, coluna="f1_weighted", ax=None):
    """Grafico de barras comparando o F1 (weighted, por padrao) dos modelos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=tabela, x="modelo", y=coluna, ax=ax, palette="rocket")
    ax.set_title("Comparacao de modelos (F1 weighted - CV)")
    ax.set_ylabel("F1 weighted")
    ax.set_ylim(0, 1)
    for i, v in enumerate(tabela[coluna].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    return ax
