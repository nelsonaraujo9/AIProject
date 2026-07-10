# Datasets

Este projeto utiliza dois datasets:

| dataset           |   rows |   features |   positive_rate |
|:------------------|-------:|-----------:|----------------:|
| Credit Card Fraud |  15492 |         30 |          0.0318 |
| Bank Marketing    |  11162 |         16 |          0.4738 |

## Ficheiros

- `bank.csv`: dataset bancário com variáveis demográficas, contactos e histórico de campanha. A variável alvo é `deposit`.
- `creditcard.csv.zip`: dataset de transações de cartão de crédito. A variável alvo é `Class`.

Para o dataset `creditcard.csv.zip`, o código usa todos os casos positivos e uma amostra fixa de 15000 casos negativos, definida por `NEGATIVE_SAMPLE_SIZE = 15000` e controlada por `RANDOM_STATE = 2074337326`, para garantir execução mais rápida e resultados reprodutíveis.
