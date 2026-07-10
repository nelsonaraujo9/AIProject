# Deteção de Fraude e Risco Bancário com Aprendizagem Supervisionada

**Unidade curricular:** Inteligência Artificial para Cibersegurança  
**Alunos:** Nelson Araújo e Ivo Pimenta  
**Números:** 20743 e a37326  
**Datasets utilizados:** `creditcard.csv.zip` e `bank.csv`  
**Modelos utilizados:** Logistic Regression, Decision Tree e Random Forest

## Resumo

Este trabalho aplica aprendizagem supervisionada a dois datasets financeiros: um dataset de transações de cartão de crédito para deteção de fraude e um dataset bancário para classificação binária de adesão/risco operacional. A abordagem compara apenas três modelos supervisionados, como solicitado: Logistic Regression, Decision Tree e Random Forest.

O melhor desempenho global foi obtido pelo modelo **Random Forest** no dataset **Credit Card Fraud**, com F1-score de **0.9170** e ROC-AUC de **0.9777**.

## 1. Introdução

A fraude financeira digital é um problema relevante em cibersegurança, especialmente em ambientes de banca online, pagamentos eletrónicos e comércio digital. Sistemas automáticos de classificação podem apoiar a identificação de transações ou perfis com maior probabilidade de risco, permitindo análise humana posterior e resposta mais rápida.

Este trabalho usa aprendizagem supervisionada para treinar modelos de classificação binária e comparar o desempenho entre dois datasets financeiros.

## 2. Datasets

Foram usados dois datasets fornecidos no projeto:

| dataset           |   rows |   features |   positive_rate |
|:------------------|-------:|-----------:|----------------:|
| Credit Card Fraud |  15492 |         30 |          0.0318 |
| Bank Marketing    |  11162 |         16 |          0.4738 |

No dataset **Credit Card Fraud**, a variável alvo é `Class`, sendo `1` fraude e `0` transação legítima. Devido ao forte desbalanceamento, o código preserva todos os casos de fraude e usa uma amostra reprodutível de transações legítimas para acelerar a execução.

No dataset **Bank Marketing**, a variável alvo é `deposit`, convertida para binária: `yes` como classe positiva e `no` como classe negativa.

## 3. Metodologia

A metodologia foi composta por seis passos: carregamento dos datasets, limpeza simples, codificação de variáveis categóricas, normalização de variáveis numéricas, divisão treino/teste estratificada e treino dos três modelos supervisionados.

Foram usados apenas três modelos:

- Logistic Regression;
- Decision Tree;
- Random Forest.

A avaliação usa accuracy, precision, recall, F1-score, ROC-AUC e Average Precision. Em problemas desbalanceados, o F1-score, o recall e a Average Precision são mais informativos do que a accuracy isolada.

## 4. Resultados

| dataset           | model               |   accuracy |   precision |   recall |   f1_score |   roc_auc |   avg_precision |
|:------------------|:--------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|
| Bank Marketing    | Random Forest       |     0.8402 |      0.7980 |   0.8873 |     0.8403 |    0.9132 |          0.8841 |
| Bank Marketing    | Logistic Regression |     0.8341 |      0.8242 |   0.8260 |     0.8251 |    0.9096 |          0.8700 |
| Bank Marketing    | Decision Tree       |     0.8255 |      0.8032 |   0.8366 |     0.8196 |    0.8668 |          0.8142 |
| Credit Card Fraud | Random Forest       |     0.9951 |      0.9906 |   0.8537 |     0.9170 |    0.9777 |          0.9199 |
| Credit Card Fraud | Decision Tree       |     0.9863 |      0.7778 |   0.7967 |     0.7871 |    0.8964 |          0.7194 |
| Credit Card Fraud | Logistic Regression |     0.9762 |      0.5812 |   0.9024 |     0.7070 |    0.9882 |          0.9233 |

## 5. Discussão

O **Random Forest** apresentou o melhor F1-score nos dois datasets. No dataset de cartão de crédito, o modelo alcançou F1-score elevado mantendo boa precisão, o que é importante para reduzir falsos positivos em cenários de fraude. A Logistic Regression teve recall elevado no dataset de fraude, mas precision inferior, o que indica maior tendência para sinalizar transações legítimas como suspeitas.

No dataset bancário, os resultados foram mais equilibrados entre os modelos, mas o Random Forest manteve a melhor combinação entre precision e recall. A Decision Tree foi competitiva, mas menos estável do que o Random Forest, por depender de uma única árvore de decisão.

## 6. Conclusão

O trabalho foi atualizado para usar dois datasets reais fornecidos e apenas três modelos supervisionados. A comparação mostra que o Random Forest é o modelo mais consistente neste conjunto de experiências. O projeto inclui código, datasets, notebook, gráficos, tabelas de resultados e relatório em PDF.

## Bibliografia

1. Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective.
2. Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems.
3. Breiman, L. (2001). Random Forests. Machine Learning.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.
