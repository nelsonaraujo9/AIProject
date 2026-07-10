# Trabalho IA4Cyber - Deteção de Fraude e Risco Bancário

## Identificação

- Alunos: Nelson Araújo e Ivo Pimenta
- Números: 20743 e a37326
- Unidade curricular: Inteligência Artificial para Cibersegurança

## Tema

Classificação supervisionada aplicada a fraude/riscos em contexto financeiro digital.

## Datasets utilizados

Foram usados dois datasets fornecidos com o projeto:

1. `creditcard.csv.zip` - transações de cartão de crédito com a classe `Class`, onde `1` representa fraude.
2. `bank.csv` - dados bancários com a classe `deposit`, convertida para classificação binária (`yes` = positivo, `no` = negativo).

No dataset de cartão de crédito foi usada uma amostra reprodutível de 15000 transações normais, mantendo todos os 492 casos de fraude, para acelerar a execução sem perder a classe minoritária.

## Modelos utilizados

Foram usados apenas 3 modelos supervisionados:

- Logistic Regression;
- Decision Tree;
- Random Forest.

## Estrutura

```text
20743_a37326/
├── data/
│   ├── bank.csv
│   ├── creditcard.csv.zip
│   └── README_dataset.md
├── figures/
│   ├── class_distribution.png
│   ├── confusion_matrix_best_model.png
│   ├── model_f1_comparison.png
│   ├── precision_recall_curve_models.png
│   └── roc_curve_models.png
├── fraud_detection_analysis.ipynb
│
├── reports/
│   ├── Relatorio_IA4Cyber_20743_a37326.pdf
│   ├── article_draft.md
│   ├── datasets_summary.csv
│   ├── results_summary.csv
│   └── results_summary.md
├── src/
│   ├── __init__.py
│   └── fraud_modeling.py
└── requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python src/fraud_modeling.py
```

O script treina os 3 modelos nos dois datasets, cria os gráficos e grava os resultados em `reports/`.

## Melhor resultado obtido

O melhor resultado pelo F1-score foi **Random Forest** no dataset **Credit Card Fraud**, com F1-score **0.9170**.


## Nota sobre reprodutibilidade

O projeto usa `RANDOM_STATE = 2074337326` para garantir que a divisão treino/teste e a amostra do dataset de fraude são sempre iguais. Por isso, ao executar novamente o script ou o notebook, os resultados devem ser reproduzidos.
