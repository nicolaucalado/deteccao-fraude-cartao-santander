# Detecção de fraude em cartão de crédito

Projeto de estudo com o conjunto Credit Card Fraud Detection. O objetivo é comparar classificadores e avaliar a classe rara (Class = 1, fraude) por recall, precisão e F1, além de visualizar curvas ROC e precisão-recall e explicar previsões com SHAP.

## Dados

O notebook carrega o CSV diretamente da URL pública da tarefa. O arquivo local creditcard.csv é usado somente para conferência e está excluído pelo .gitignore; não o envie ao GitHub (tem aproximadamente 151 MB).

Na cópia local conferida: 284.807 transações, 31 colunas, 284.315 normais e 492 fraudes (0,1727%); não foram encontrados valores ausentes. Time é o número de segundos decorridos desde a primeira transação. V1 a V28 são componentes PCA anonimizados; Amount recebe log1p; Class é o alvo binário.

## Método

- Divisão treino/teste estratificada (80/20), com estado aleatório fixo.
- Regressão logística com StandardScaler em Pipeline e pesos balanceados.
- Random Forest com class_weight="balanced" e XGBoost com scale_pos_weight.
- Comparação de recall, precisão, F1, ROC AUC e average precision; relatório, matriz de confusão e curvas ROC/PR.
- Comparação de limiares 0,5 e 0,3; busca pequena de C da regressão logística com GridSearchCV selecionando pelo recall.
- Importância global da Random Forest e explicações SHAP para o XGBoost.

As tabelas e figuras são produzidas pelas células do notebook. Execute no Colab e mantenha as saídas salvas; use os valores gerados ali no portfólio, pois resultados podem variar com as versões das bibliotecas.

## Resultados do teste
Com divisão estratificada fixa (42), o conjunto de teste tem 56.962 transações, incluindo 98 fraudes. As métricas abaixo são da classe fraude, arredondadas para duas casas percentuais; os valores completos e relatórios estão salvos no notebook.

| Modelo | Recall | Precisão | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regressão logística | 91,84% | 6,03% | 11,31% | 97,11% | 71,13% |
| Random Forest | 78,57% | 92,77% | 85,08% | 96,13% | 86,70% |
| XGBoost | 86,73% | 69,11% | 76,92% | 98,41% | 86,44% |
| Dummy (sempre normal) | 0,00% | 0,00% | 0,00% | 50,00% | 0,17% |

O Dummy tem acurácia aproximada de 99,83%, mas encontra zero das 98 fraudes do teste: esse é o motivo para não escolher modelo por acurácia. A regressão logística encontra mais fraudes, porém gera muitos falsos alertas; a Random Forest tem a maior precisão e F1 entre esses modelos.

### Limiar e ajuste
No modelo de regressão logística, o limiar 0,5 detectou 90/98 fraudes (recall 91,84%), com precisão 6,03%, F1 11,31% e 1.493 alertas. Com limiar 0,3, o recall permaneceu 91,84%, mas a precisão caiu para 2,62%, o F1 para 5,10% e o volume subiu para 3.432 alertas. Neste teste, baixar o limiar não capturou fraudes adicionais.

O GridSearchCV escolheu C = 0,1, com recall médio 89,84% na validação cruzada de treino. No teste, o modelo ajustado a limiar 0,3 detectou 90 fraudes e produziu 3.412 alertas, dos quais 3.322 eram falsos positivos. O valor de 0,3 segue o roteiro do desafio; uma escolha operacional exigiria custos e capacidade de revisão.

### O que a explicação SHAP indicou
Na amostra com as 98 fraudes do teste e 1.000 transações normais, os maiores valores médios absolutos de SHAP foram V14 (2,043), V4 (1,530), V10 (0,674), V12 (0,659) e V8 (0,485). A célula waterfall mostra uma fraude real (linha 8.972) com probabilidade estimada de 0,817. Como V1–V28 são componentes PCA anonimizados, essa importância não diz qual comportamento original da compra causou o alerta; SHAP descreve a saída do modelo.

### O que foi acrescentado ao roteiro
Além da comparação dos três modelos sugeridos, incluí um Dummy para tornar visível a falha da acurácia, PR AUC junto com as métricas da classe fraude, e uma tabela com o volume de alertas por limiar. A amostra SHAP inclui explicitamente as fraudes do teste para garantir um exemplo local. O README registra que 0,3 não elevou o recall neste split. Não atribuo diferenças específicas à aula, pois o notebook original da Expert não foi fornecido.

## Como executar

Abra deteccao_fraude_cartao.ipynb no Google Colab e execute as células em ordem. Ou instale as dependências listadas em requirements.txt e use Jupyter com acesso à internet para buscar o CSV.

## Interpretação e limitações

Acurácia isolada é inadequada: prever tudo como normal já acertaria cerca de 99,83% dos registros, mas teria recall zero para fraude. Recall mede a fração de fraudes detectadas; precisão, a fração dos alertas que realmente eram fraude; F1 equilibra ambos. Limiar menor pode recuperar mais fraudes e gerar mais falsos positivos, então a escolha depende de capacidade e custo operacional.

SHAP explica contribuições para as decisões do modelo, mas os componentes PCA não têm interpretação direta de negócio e explicação não é causalidade. A divisão aleatória estratificada segue o briefing; um corte temporal seria uma avaliação adicional importante.

## Referências

- [CSV público indicado no desafio](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
- [Página do dataset no Kaggle (ULB/Worldline)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [XGBoost: parâmetros](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [SHAP TreeExplainer: a saída bruta para XGBoost binário corresponde a log-odds](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
