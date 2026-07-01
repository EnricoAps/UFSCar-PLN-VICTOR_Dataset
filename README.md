# Categorização Automática de Documentos Jurídicos (VICTOR/STF)

Classificação de peças processuais do STF em 5 classes (0-Acórdão, 1-ARE,
2-Despacho, 3-RE, 4-Sentença). Métrica da competição: **F1-Score**.

## Estrutura

```
main.ipynb                      # notebook principal (4 seções)
scripts/preprocessamento.py     # carga, correção de encoding, limpeza, stopwords, stemming
scripts/analise_exploratoria.py # distribuição, tamanhos, vocabulário
scripts/experimentos.py         # TF-IDF (palavra+char), modelos clássicos, CV, submissão
scripts/analise_resultados.py   # relatório, matriz de confusão, F1 por classe
scripts/bertimbau_kaggle.py     # fine-tuning do BERTimbau (rodar no Kaggle/GPU)
train.csv / test.csv            # base (NÃO incluir no .zip de entrega)
```

## Como rodar localmente (baseline clássico, CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['stopwords','rslp']]"
jupyter notebook main.ipynb     # executar as células de cima para baixo
```

Gera `submission.csv` no formato `Id,Category`.

## Como rodar o BERTimbau (Kaggle, GPU)

1. Crie um Notebook na página da competição (Accelerator = GPU).
2. Anexe os scripts ou cole `scripts/bertimbau_kaggle.py`.
3. Ajuste `CAMINHO_TREINO`/`CAMINHO_TESTE` para `/kaggle/input/<competicao>/...`.
4. Execute — gera `/kaggle/working/submission.csv`.

## Decisões de projeto

- **Classe `-1`**: páginas não rotuladas do VICTOR — removidas do treino supervisionado.
- **Desbalanceamento** (classe 3 ~63%): `class_weight='balanced'` / perda ponderada.
- **Encoding**: mojibake corrigido com `.encode('latin-1').decode('utf-8')`.
- **Representação**: TF-IDF palavra (1-2) + caractere (3-5), robusto a ruído de OCR.
