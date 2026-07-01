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

# Fine-tuning do BERTimbau em VALIDACAO CRUZADA (K-FOLD) para a classificacao
# de pecas juridicas (VICTOR/STF).
#
# REQUER GPU (detecta o ambiente automaticamente: Kaggle ou local com CUDA).
# Em um unico run, treina N_FOLDS modelos e gera:
#   - {TAG}_oof_train.csv : probabilidades OUT-OF-FOLD (HONESTAS) p/ todo o treino,
#                           usadas para tunar os pesos do ensemble sem vies.
#   - {TAG}_probs_test.csv : probabilidades no teste, MEDIADAS sobre os N folds.
# Tempo estimado: ~5-6h (5 folds x 4 epocas, MAX_LEN=512).

import gc
import os
import re

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------------
MODELO = "neuralmind/bert-base-portuguese-cased"
TAG = "bertimbau"          # nome dos arquivos de saida
MAX_LEN = 512
HEAD_RATIO = 0.25          # truncamento head+tail (inicio + fim da peca)
N_CLASSES = 5
N_FOLDS = 5
SEED = 42
EPOCAS = 4
BATCH = 8
GRAD_ACCUM = 2
LR = 2e-5

# Deteccao automatica do ambiente (Kaggle vs local).
if os.path.isdir("/kaggle/input"):
    CAMINHO_TREINO = "/kaggle/input/competitions/ufscar-pln2026-pf/train.csv"
    CAMINHO_TESTE = "/kaggle/input/competitions/ufscar-pln2026-pf/test.csv"
    DIR_SAIDA = "/kaggle/working"
else:
    CAMINHO_TREINO = "train.csv"
    CAMINHO_TESTE = "test.csv"
    DIR_SAIDA = "."

torch.manual_seed(SEED)
np.random.seed(SEED)

_WRAP = re.compile(r'^\s*\{\s*"|"\s*\}\s*$')


def corrigir_encoding(t):
    if not isinstance(t, str):
        return ""
    try:
        return t.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t


def limpeza_leve(t):
    t = corrigir_encoding(t)
    t = _WRAP.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


class DatasetJuridico(Dataset):
    """Tokeniza com truncamento head+tail para textos longos."""

    def __init__(self, textos, rotulos, tokenizer, max_len, head_ratio=HEAD_RATIO):
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        limite = max_len - 2
        head = int(limite * head_ratio)
        tail = limite - head
        brutos = tokenizer(list(textos), add_special_tokens=False,
                           truncation=False)["input_ids"]
        self.input_ids = []
        for ids in brutos:
            if len(ids) > limite:
                ids = ids[:head] + ids[-tail:]
            self.input_ids.append([self.cls_id] + ids + [self.sep_id])
        self.rotulos = list(rotulos) if rotulos is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        ids = self.input_ids[i]
        item = {"input_ids": torch.tensor(ids),
                "attention_mask": torch.ones(len(ids), dtype=torch.long)}
        if self.rotulos is not None:
            item["labels"] = torch.tensor(self.rotulos[i])
        return item


def metricas(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


class TrainerPonderado(Trainer):
    def __init__(self, pesos_classe=None, **kwargs):
        super().__init__(**kwargs)
        self.pesos_classe = pesos_classe

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.pesos_classe)
        loss = loss_fct(outputs.logits.view(-1, N_CLASSES), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _softmax_np(logits):
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Modelo:", MODELO, "| Folds:", N_FOLDS)

    treino = pd.read_csv(CAMINHO_TREINO)
    treino["Body"] = treino["Body"].map(limpeza_leve)
    treino = treino[treino["Category"] != -1].reset_index(drop=True)
    teste = pd.read_csv(CAMINHO_TESTE)
    teste["Body"] = teste["Body"].map(limpeza_leve)

    X = treino["Body"].values
    y = treino["Category"].values
    ids = treino["Id"].values

    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    ds_test = DatasetJuridico(teste["Body"].values, None, tokenizer, MAX_LEN)

    oof = np.zeros((len(treino), N_CLASSES))           # probs out-of-fold
    test_probs = np.zeros((len(teste), N_CLASSES))     # media sobre os folds

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== FOLD {fold + 1}/{N_FOLDS} =====")
        ds_tr = DatasetJuridico(X[tr_idx], y[tr_idx], tokenizer, MAX_LEN)
        ds_va = DatasetJuridico(X[va_idx], y[va_idx], tokenizer, MAX_LEN)

        pesos = compute_class_weight("balanced", classes=np.arange(N_CLASSES),
                                     y=y[tr_idx])
        pesos = torch.tensor(pesos, dtype=torch.float).to(device)

        modelo = AutoModelForSequenceClassification.from_pretrained(
            MODELO, num_labels=N_CLASSES)

        args = TrainingArguments(
            output_dir=os.path.join(DIR_SAIDA, f"{TAG}_fold{fold}"),
            num_train_epochs=EPOCAS,
            per_device_train_batch_size=BATCH,
            per_device_eval_batch_size=BATCH * 2,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_steps=200,
            report_to="none",
            seed=SEED,
        )
        trainer = TrainerPonderado(
            pesos_classe=pesos, model=modelo, args=args,
            train_dataset=ds_tr, eval_dataset=ds_va,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=metricas,
        )
        trainer.train()

        oof[va_idx] = _softmax_np(trainer.predict(ds_va).predictions)
        test_probs += _softmax_np(trainer.predict(ds_test).predictions) / N_FOLDS
        f_fold = f1_score(y[va_idx], oof[va_idx].argmax(1), average="weighted")
        print(f"[FOLD {fold + 1}] OOF F1-weighted = {f_fold:.4f}")

        # libera memoria da GPU antes do proximo fold
        del trainer, modelo
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Salva OOF e probs de teste (mediadas) ----
    pd.DataFrame(oof, columns=[f"p{c}" for c in range(N_CLASSES)]) \
        .assign(Id=ids, true=y) \
        .to_csv(os.path.join(DIR_SAIDA, f"{TAG}_oof_train.csv"), index=False)
    pd.DataFrame(test_probs, columns=[f"p{c}" for c in range(N_CLASSES)]) \
        .assign(Id=teste["Id"].values) \
        .to_csv(os.path.join(DIR_SAIDA, f"{TAG}_probs_test.csv"), index=False)

    wo = f1_score(y, oof.argmax(1), average="weighted")
    mo = f1_score(y, oof.argmax(1), average="macro")
    print(f"\n>>> OOF GLOBAL {TAG}: F1-weighted={wo:.4f}  F1-macro={mo:.4f}")

    # submissao do BERT k-fold sozinho (referencia)
    pd.DataFrame({"Id": teste["Id"].values, "Category": test_probs.argmax(1)}) \
        .to_csv(os.path.join(DIR_SAIDA, "submission.csv"), index=False)
    print(f"Salvos: {TAG}_oof_train.csv, {TAG}_probs_test.csv, submission.csv")


if __name__ == "__main__":
    main()
