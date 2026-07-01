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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

# F1 ponderado (weighted)
SCORING = "f1_weighted"
SEED = 42

def construir_vetorizador(palavra=True, char=True, max_features_palavra=50000,
                          max_features_char=50000):
    """Cria um vetorizador TF-IDF combinando n-gramas de palavra e de caractere.

    A combinacao palavra + caractere costuma ser robusta a ruido de OCR, pois
    os n-gramas de caractere capturam radicais mesmo com erros de digitacao.
    """
    transformadores = []
    if palavra:
        transformadores.append((
            "palavra",
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                            sublinear_tf=True, min_df=3,
                            max_features=max_features_palavra),
            "texto_limpo",
        ))
    if char:
        transformadores.append((
            "char",
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            sublinear_tf=True, min_df=3,
                            max_features=max_features_char),
            "texto_limpo",
        ))
    return ColumnTransformer(transformadores)

def construir_modelos():
    """Retorna um dicionario {nome: estimador} com os modelos classicos."""
    return {
        "NaiveBayes": ComplementNB(),
        "LogReg": LogisticRegression(max_iter=2000, C=5.0,
                                     class_weight="balanced",
                                     random_state=SEED),
        # LinearSVC nao fornece predict_proba; calibramos para ter probabilidades
        # (necessarias para o ensemble). C=2.0 escolhido por grid search (F1-weighted).
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(C=2.0, class_weight="balanced", random_state=SEED),
            cv=3,
        ),
    }


def montar_pipeline(modelo, **kwargs_vetorizador):
    """Encapsula vetorizador + modelo em um Pipeline do scikit-learn."""
    return Pipeline([
        ("tfidf", construir_vetorizador(**kwargs_vetorizador)),
        ("clf", modelo),
    ])

def ajustar_hiperparametros(X, y, n_splits=5):
    """Busca em grade sobre C do LinearSVC, avaliada por F1-weighted em CV.

    Testa C in [0.5, 1.0, 2.0, 5.0] e imprime o ranking completo.
    Retorna o melhor valor de C encontrado.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pipe = montar_pipeline(
        CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000), cv=3
        )
    )
    param_grid = {"clf__estimator__C": [0.5, 1.0, 2.0, 5.0]}
    gs = GridSearchCV(pipe, param_grid, scoring="f1_weighted",
                      cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X, y)

    resultados = pd.DataFrame({
        "C": gs.cv_results_["param_clf__estimator__C"],
        "F1-weighted": gs.cv_results_["mean_test_score"],
        "std": gs.cv_results_["std_test_score"],
    }).sort_values("F1-weighted", ascending=False).reset_index(drop=True)
    print(resultados.to_string(index=False))
    print(f"\nMelhor C = {gs.best_params_['clf__estimator__C']}  "
          f"(F1-weighted = {gs.best_score_:.4f})")
    return gs.best_params_["clf__estimator__C"]

def avaliar_cv(pipeline, X, y, n_splits=5):
    """Validacao cruzada estratificada; retorna predicoes out-of-fold (OOF) e
    o F1 weighted e macro.

    Usa cross_val_predict para obter uma predicao para cada amostra (gerada
    quando ela estava no fold de teste), permitindo relatorio e matriz de
    confusao honestos sobre todo o conjunto de treino.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    f1_w = f1_score(y, y_pred, average="weighted")
    f1_m = f1_score(y, y_pred, average="macro")
    return y_pred, f1_w, f1_m


def comparar_modelos(X, y, n_splits=5, **kwargs_vetorizador):
    """Avalia todos os modelos classicos por CV e retorna um DataFrame resumo.

    A ordenacao usa o F1 weighted (metrica oficial); o F1 macro tambem e
    reportado por ser robusto ao desbalanceamento.
    """
    resultados = []
    predicoes = {}
    for nome, modelo in construir_modelos().items():
        pipe = montar_pipeline(modelo, **kwargs_vetorizador)
        y_pred, f1_w, f1_m = avaliar_cv(pipe, X, y, n_splits)
        resultados.append({"modelo": nome, "f1_weighted": f1_w, "f1_macro": f1_m})
        predicoes[nome] = y_pred
        print(f"{nome:<12} F1-weighted (CV) = {f1_w:.4f}   F1-macro = {f1_m:.4f}")
    tabela = pd.DataFrame(resultados).sort_values("f1_weighted", ascending=False)
    return tabela.reset_index(drop=True), predicoes

def treinar_final(pipeline, X, y):
    """Treina o pipeline em todo o conjunto de treino rotulado."""
    pipeline.fit(X, y)
    return pipeline


def gerar_submissao(pipeline, df_teste, caminho_saida="submission.csv",
                    coluna_id="Id"):
    """Gera o arquivo de submissao no formato Id,Category exigido pelo Kaggle."""
    preds = pipeline.predict(df_teste)
    sub = pd.DataFrame({coluna_id: df_teste[coluna_id].values,
                        "Category": preds})
    sub.to_csv(caminho_saida, index=False)
    print(f"Submissao salva em '{caminho_saida}' ({len(sub)} linhas).")
    return sub
