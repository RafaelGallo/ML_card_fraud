# Projeto machine learning - Indentificação fraude cartão de crédito

[![author](https://img.shields.io/badge/author-Rafael_Gallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html) 
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-green.svg?style=flat)]([https://github.com/danielesantiago/Data-Science](https://github.com/RafaelGallo?tab=repositories))
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://tensorflow.org/stable/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Cuda-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/GPU-green.svg)](https://scikit-learn.org/stable/)

![](https://github.com/RafaelGallo/ML_card_fraud/blob/main/img/SL-110820-37810-04.jpg)

# Projeto 1 - Indentificação fraude cartão de crédito - Ieee-fraud-detection

## Descrição projeto
Imagine estar no caixa do supermercado com uma longa fila atrás de você, e o caixa anuncia em voz alta que seu cartão foi recusado. Nesse momento, provavelmente você não está pensando na ciência de dados que determinou seu destino. Envergonhado e certo de que tem fundos para cobrir tudo o ue é necessário para uma festa épica de nachos para 50 dos seus amigos mais próximos, você tenta passar o cartão novamente. Mesmo resultado. Enquanto você se afasta e permite que o caixa atenda o próximo cliente, você recebe uma mensagem de texto do seu banco. "Aperte 1 se você realmente tentou gastar $500 em queijo cheddar." Embora possa ser inconveniente (e muitas vezes embaraçoso) no momento, esse sistema de prevenção de fraudes está economizando milhões de dólares por ano para os consumidores. Pesquisadores da Sociedade de Inteligência Computacional do IEEE (IEEE-CIS) desejam melhorar esse número, ao mesmo tempo em que melhoram a experiência do cliente. Com uma detecção de fraudes de maior precisão, você pode aproveitar seus chips sem complicações. A IEEE-CIS trabalha em várias áreas de inteligência artificial e aprendizado de máquina, incluindo redes neurais profundas, sistemas difusos, computação evolutiva e inteligência de enxame. Hoje, eles estão se associando à principal empresa de serviços de pagamento do mundo, a Vesta Corporation, em busca das melhores soluções para a indústria de prevenção de fraudes, e agora você está convidado a participar do desafio. Nesta competição, você avaliará modelos de aprendizado de máquina em um grande conjunto de dados desafiador. Os dados provêm de transações de comércio eletrônico do mundo real da Vesta e contêm uma ampla gama de recursos, desde o tipo de dispositivo até as características do produto. Você também tem a oportunidade de criar novos recursos para melhorar seus resultados. Se for bem-sucedido, você melhorará a eficácia dos alertas de transações fraudulentas para milhões de pessoas em todo o mundo, ajudando centenas de milhares de empresas a reduzir suas perdas com fraudes e aumentar sua receita. E, é claro, você salvará pessoas de festa, assim como você, do incômodo de falsos positivos.
A Vesta Corporation forneceu o conjunto de dados para esta competição. A Vesta Corporation é a precursora em soluções de pagamento de comércio eletrônico garantidas. Fundada em 1995, a Vesta foi pioneira no processo de transações de pagamento de cartão não presente totalmente garantidas para a indústria de telecomunicações. Desde então, a Vesta expandiu firmemente suas capacidades de ciência de dados e aprendizado de máquina em todo o mundo e consolidou sua posição como líder em pagamentos de comércio eletrônico garantidos. Hoje, a Vesta garante mais de $18 bilhões em transações anualmente. Foto do cabeçalho de Tim Evans no Unsplash.

## Descrição do Conjunto de Dados

Nesta competição, você está prevendo a probabilidade de que uma transação online seja fraudulenta, conforme indicado pelo alvo binário isFraud. Os dados estão divididos em dois arquivos, identity (identidade) e transaction (transação), que são unidos pelo TransactionID. Nem todas as transações têm informações de identidade correspondentes.

## Recursos categóricos - transação

- ProductCD
- card1 - card6
- caddr1 addr2
- P_emaildomain
- R_emaildomain
- M1 - M9

## Características Categóricas - Identidade

- DeviceType
- DeviceInfo
- id_12 - id_38

O recurso TransactionDT é um timedelta de uma determinada data e hora de referência (não um carimbo de data/hora real). Você pode ler mais sobre os dados nesta postagem do anfitrião da competição.

## Arquivos

- train_{transaction, Identity}.csv - o conjunto de treinamento
- test_{transaction, Identity}.csv - o conjunto de teste (você deve prever o valor isFraud para essas observações
- sample_submission.csv - um arquivo de envio de amostra no formato correto

## Citação
@ Addison Howard, Bernadette Bouchon-Meunier, IEEE CIS, inversão, John Lei, Lynn@Vesta, Marcus2010, Prof. Hussein Abbass. (2019). IEEE-CIS Fraud Detection. Kaggle.
https://kaggle.com/competitions/ieee-fraud-detection

## Base dados
https://kaggle.com/competitions/ieee-fraud-detection

## Stack utilizada

**Programação** Python

**Machine learning**: Scikit-learn.

**Leitura CSV**: Pandas.

**Análise de dados**: Seaborn, Matplotlib.

**Modelo machine learning**: Random Forest


## Variáveis de Ambiente

Para criar e gerenciar ambientes virtuais (env) no Windows, você pode usar a ferramenta venv, que vem com as versões mais recentes do Python (3.3 e posteriores). Aqui está um guia passo a passo de como criar um ambiente virtual no Windows:

1 Abrir o Prompt de Comando: Pressione a tecla Win + R para abrir o "Executar", digite cmd e pressione Enter. Isso abrirá o Prompt de Comando do Windows.

2 Navegar até a pasta onde você deseja criar o ambiente virtual: Use o comando cd para navegar até a pasta onde você deseja criar o ambiente virtual. Por exemplo:

`cd caminho\para\sua\pasta`

3 Criar o ambiente virtual: Use o comando python -m venv nome_do_seu_env para criar o ambiente virtual. Substitua nome_do_seu_env pelo nome que você deseja dar ao ambiente. 

`python -m venv myenv`

4 Ativar o ambiente virtual: No mesmo Prompt de Comando, você pode ativar o ambiente virtual usando o script localizado na pasta Scripts dentro do ambiente virtual. Use o seguinte comando:

`nome_do_seu_env\Scripts\activate`

5 Desativar o ambiente virtual: Para desativar o ambiente virtual quando você não precisar mais dele, simplesmente digite deactivate e pressione Enter no Prompt de Comando.

`myenv\Scripts\activate`

Obs: Lembre-se de que, uma vez que o ambiente virtual está ativado, todas as instalações de pacotes usando pip serão isoladas dentro desse ambiente, o que é útil para evitar conflitos entre diferentes projetos. Para sair completamente do ambiente virtual, feche o Prompt de Comando ou digite deactivate. Lembre-se também de que essas instruções pressupõem que o Python esteja configurado corretamente em seu sistema e acessível pelo comando python. Certifique-se de ter o Python instalado e adicionado ao seu PATH do sistema. Caso prefira, você também pode usar ferramentas como o Anaconda, que oferece uma maneira mais abrangente de gerenciar ambientes virtuais e pacotes em ambientes de desenvolvimento Python.

## Pacote no anaconda
Para instalar um pacote no Anaconda, você pode usar o gerenciador de pacotes conda ou o gerenciador de pacotes Python padrão, pip, que também está disponível dentro dos ambientes do Anaconda. Aqui estão as etapas para instalar um pacote usando ambos os métodos:

## Usando o Conda
1 Abra o Anaconda Navigator ou o Anaconda Prompt, dependendo da sua preferência.

2 Para criar um novo ambiente (opcional, mas recomendado), você pode executar o seguinte comando no Anaconda Prompt (substitua nome_do_seu_ambiente pelo nome que você deseja dar ao ambiente):

`conda create --name nome_do_seu_ambiente`

3 Ative o ambiente recém-criado com o seguinte comando (substitua nome_do_seu_ambiente pelo nome do ambiente que você criou):

`conda create --name nome_do_seu_ambiente`

4 Para instalar um pacote específico, use o seguinte comando (substitua nome_do_pacote pelo nome do pacote que você deseja instalar):

`conda install nome_do_pacote`


## Instalação
Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo modelo machine learning  

```
# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score

# Criando dados de exemplo (você pode substituir pelos seus próprios dados)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calculando e exibindo a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Calculando e exibindo a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

# Exibindo o relatório de classificação
class_report = classification_report(y_test, y_pred)
print("Relatório de Classificação:")
print(class_report)

# Calculando probabilidades para a curva ROC
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculando a área sob a curva ROC (AUC)
auc = roc_auc_score(y_test, y_probs)
print(f"AUC: {auc:.2f}")

# Plotando a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

# Importando biblioteca 
import joblib

# Após treinar o modelo (model.fit(X_train, y_train))
# Salvar o modelo em um arquivo .pkl
joblib.dump(model, 'modelo_random_forest.pkl')
```
