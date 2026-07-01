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

# Fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased) para a
# classificacao de pecas juridicas (VICTOR/STF).
#
# REQUER GPU. O script detecta automaticamente o ambiente:
#   - Local: le train.csv/test.csv do diretorio atual e usa CUDA se disponivel
#     (cai para CPU, porem MUITO lento); saidas gravadas no diretorio atual.
#   - Kaggle: le de /kaggle/input/... e grava em /kaggle/working/.
# Gera submission.csv e os arquivos de probabilidades {TAG}_probs_test.csv e
# {TAG}_val_probs.csv (usados pelo ensemble).

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
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
# Escolha do transformer. Rode UMA VEZ por modelo (os outputs sao nomeados por
# TAG, entao nao se sobrescrevem e podem ser combinados no ensemble depois):
#   BERTimbau (geral)   : "neuralmind/bert-base-portuguese-cased"          -> TAG "bertimbau"
#   LegalBERT-pt (STF)  : "dominguesm/legal-bert-base-cased-ptbr"          -> TAG "legalbert"
MODELO = "neuralmind/bert-base-portuguese-cased"
TAG = "bertimbau"      # MUDE para "legalbert" ao rodar o LegalBERT-pt
MAX_LEN = 512          # janela maxima do BERT; captura mais do documento longo
N_CLASSES = 5
SEED = 42
EPOCAS = 4
BATCH = 8              # 512 tokens: batch menor p/ caber na GPU (T4/P100)
GRAD_ACCUM = 2         # batch efetivo = BATCH * GRAD_ACCUM = 16
LR = 2e-5
# Truncamento "head+tail": fracao de tokens mantida do INICIO; o resto vem do
# FIM da peca (ementa/dispositivo costumam estar no final). Sun et al. (2019).
HEAD_RATIO = 0.25

# Deteccao automatica do ambiente: usa os caminhos do Kaggle se existirem,
# caso contrario le do diretorio local (onde estao train.csv e test.csv).
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


# ----------------------------------------------------------------------------
# Pre-processamento LEVE (BERT lida bem com texto bruto; so corrigimos encoding
# e removemos o invlucro {"..."}). NAO removemos stopwords nem acentos aqui.
# ----------------------------------------------------------------------------
import re

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


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class DatasetJuridico(Dataset):
    """Tokeniza com truncamento head+tail para textos longos.

    Documentos maiores que max_len mantem os `head` primeiros tokens e os
    ultimos (max_len - 2 - head), preservando [CLS] no inicio e [SEP] no fim.
    O padding fica a cargo do DataCollatorWithPadding.
    """

    def __init__(self, textos, rotulos, tokenizer, max_len, head_ratio=HEAD_RATIO):
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_len = max_len
        limite = max_len - 2                      # reserva p/ [CLS] e [SEP]
        head = int(limite * head_ratio)
        tail = limite - head
        # Tokeniza sem special tokens e sem truncar para acessar a sequencia toda.
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


# Trainer com perda ponderada (lida com o desbalanceamento das classes).
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- Dados ----
    treino = pd.read_csv(CAMINHO_TREINO)
    treino["Body"] = treino["Body"].map(limpeza_leve)
    treino = treino[treino["Category"] != -1].reset_index(drop=True)

    teste = pd.read_csv(CAMINHO_TESTE)
    teste["Body"] = teste["Body"].map(limpeza_leve)

    X_tr, X_val, y_tr, y_val, id_tr, id_val = train_test_split(
        treino["Body"].values, treino["Category"].values, treino["Id"].values,
        test_size=0.1, stratify=treino["Category"].values, random_state=SEED,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    ds_tr = DatasetJuridico(X_tr, y_tr, tokenizer, MAX_LEN)
    ds_val = DatasetJuridico(X_val, y_val, tokenizer, MAX_LEN)
    ds_test = DatasetJuridico(teste["Body"].values, None, tokenizer, MAX_LEN)

    pesos = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_tr)
    pesos = torch.tensor(pesos, dtype=torch.float).to(device)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO, num_labels=N_CLASSES
    )

    args = TrainingArguments(
        output_dir=os.path.join(DIR_SAIDA, f"{TAG}_ckpt"),
        num_train_epochs=EPOCAS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    trainer = TrainerPonderado(
        pesos_classe=pesos,
        model=modelo,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metricas,
    )

    trainer.train()
    print("Validacao:", trainer.evaluate())

    # ---- Probabilidades na VALIDACAO (para decidir/tunar o ensemble offline) ----
    val_logits = trainer.predict(ds_val).predictions
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    df_val = pd.DataFrame(val_probs, columns=[f"p{c}" for c in range(N_CLASSES)])
    df_val["Id"] = id_val
    df_val["true"] = y_val
    df_val.to_csv(os.path.join(DIR_SAIDA, f"{TAG}_val_probs.csv"), index=False)
    wv = f1_score(y_val, val_probs.argmax(1), average="weighted")
    mv = f1_score(y_val, val_probs.argmax(1), average="macro")
    print(f"[VAL] {TAG} F1-weighted={wv:.4f}  F1-macro={mv:.4f}")

    # ---- Predicao no teste e submissao ----
    logits = trainer.predict(ds_test).predictions
    # Probabilidades (softmax) salvas para permitir ENSEMBLE depois.
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pd.DataFrame(probs, columns=[f"p{c}" for c in range(N_CLASSES)]) \
        .assign(Id=teste["Id"].values) \
        .to_csv(os.path.join(DIR_SAIDA, f"{TAG}_probs_test.csv"), index=False)

    preds = np.argmax(probs, axis=-1)
    sub = pd.DataFrame({"Id": teste["Id"].values, "Category": preds})
    sub.to_csv(os.path.join(DIR_SAIDA, "submission.csv"), index=False)
    print(f"submission.csv + {TAG}_probs_test.csv + {TAG}_val_probs.csv salvos.")


if __name__ == "__main__":
    main()
