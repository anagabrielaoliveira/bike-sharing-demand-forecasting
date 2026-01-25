# Exercício: Previsão de Demanda de Bicicletas Compartilhadas

## 1. Sumário das Alterações

Este projeto implementa um pipeline completo de Machine Learning para previsão de demanda de bicicletas compartilhadas usando **Metaflow** como orquestrador de workflows. O pipeline foi estruturado de forma modular e segue boas práticas de engenharia de ML.

### Bibliotecas Utilizadas:

- **Metaflow** (>=2.19.17): Framework para orquestração de workflows de ML, permitindo versionamento e reprodutibilidade
- **Pandas** (>=3.0.0): Manipulação e análise de dados
- **Scikit-learn** (>=1.8.0): Modelos de ML (Random Forest) e métricas de avaliação
- **LightGBM** (>=4.6.0): Modelo de gradient boosting otimizado
- **XGBoost**: Modelo de gradient boosting (via scikit-learn MultiOutputRegressor)
- **Optuna** (>=4.7.0): Framework para otimização de hiperparâmetros usando técnicas avançadas como TPE (Tree-structured Parzen Estimator)
- **Seaborn** (>=0.13.2): Visualização de dados

### Principais Alterações:

1. **Estruturação Modular**: Código organizado em módulos (`preprocess`, `train`, `hpo`, `eval`)
2. **Pipeline Metaflow**: Implementação de um flow completo com etapas de pré-processamento, treinamento, otimização e avaliação
3. **Remoção de Outliers**: Implementação de múltiplas técnicas de detecção e remoção de outliers
4. **Feature Engineering**: Criação de features temporais, interações e features de pico
5. **Hyperparameter Optimization**: Otimização automática de hiperparâmetros para 3 modelos diferentes
6. **Avaliação Comparativa**: Sistema para comparar e selecionar o melhor modelo

## 2. Diagrama do Flow

```
┌─────────┐
│  START  │  Carrega dados de treino (data/train.csv)
└────┬────┘
     │
     ▼
┌──────────────┐
│  PREPROCESS  │  Cria features temporais, interações e features de pico
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ OUTLIER_REMOVAL  │  Remove outliers usando múltiplas técnicas
└──────┬───────────┘
       │
       ▼
┌─────────────────────┐
│ TRAIN_BASELINE_MODEL│  Treina modelo Random Forest baseline
└──────┬──────────────┘
       │
       ▼
┌─────────┐
│   HPO   │  Otimiza hiperparâmetros para:
└────┬────┘    - LightGBM
     │         - XGBoost
     │         - Random Forest
     │
     ▼
┌──────────────────┐
│ EVALUATE_MODELS  │  Compara todos os modelos e seleciona o melhor
└──────┬───────────┘
       │
       ▼
┌─────────┐
│   END   │
└─────────┘
```

## 3. Classes Construídas

### `OutlierRemoval` (`src/preprocess/outlier_removal.py`)
Classe responsável pela remoção de outliers nos dados de treino. Implementa um padrão builder/fluent interface que permite encadear múltiplas operações de remoção:
- **temp_and_atemp_outlier_removal()**: Remove outliers baseado em limites de temperatura e temperatura aparente
- **humidity_remove_zero_values()**: Remove valores zero de umidade
- **temp_vs_windspeed_outlier_removal()**: Remove outliers baseado na interação temperatura vs velocidade do vento
- **train_if_model()**: Treina um modelo Isolation Forest para detecção de outliers
- **if_removal_predict()**: Usa o Isolation Forest treinado para remover outliers
- **get_dataframe()**: Retorna o dataframe processado

### `OptunaHPO` (`src/hpo/optuna_hpo.py`)
Classe responsável pela otimização de hiperparâmetros usando Optuna. Implementa otimização para três modelos diferentes:
- **lgbm_optimize()**: Otimiza hiperparâmetros do LightGBM usando TPE sampler e Hyperband pruner
- **xgb_optimize()**: Otimiza hiperparâmetros do XGBoost
- **rf_optimize()**: Otimiza hiperparâmetros do Random Forest
- **\_create_study()**: Cria um estudo Optuna com configurações otimizadas
- **\_save_study_results()**: Salva os resultados da otimização em arquivo JSON

### `EvalResults` (`src/eval/eval_results.py`)
Classe responsável por avaliar e comparar os resultados de todos os modelos (baseline + HPO):
- **\_load_results()**: Carrega todos os resultados de HPO e baseline
- **\_return_best_model()**: Compara todos os modelos e retorna o melhor baseado no menor MAPE
- **evaluate_models()**: Avalia e compara todos os modelos e retorna o melhor baseado no menor MAPE

### `BikeSharingDemandFlow` (`src/flow.py`)
Classe principal que herda de `FlowSpec` do Metaflow e define o pipeline completo:
- **start**: Etapa inicial que carrega os dados
- **preprocess**: Cria features temporais e de interação
- **outlier_removal**: Remove outliers dos dados
- **train_baseline_model**: Treina modelo baseline Random Forest
- **hpo**: Executa otimização de hiperparâmetros para 3 modelos
- **evaluate_models**: Compara e seleciona o melhor modelo
- **end**: Etapa final do pipeline

## 4. Instalação

### Pré-requisitos

Este projeto utiliza **uv**, um gerenciador de pacotes Python moderno e rápido desenvolvido pela Astral (mesmos criadores do ruff). O uv é uma alternativa mais rápida ao pip e pip-tools.

### Instalando o uv

**No macOS e Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Ou usando Homebrew (macOS):**
```bash
brew install uv
```

**No Windows:**

Você pode instalar o uv de duas formas:

**Opção 1: Usando PowerShell**
1. Abra o **PowerShell** como Administrador (clique com botão direito e selecione "Executar como administrador")
2. Execute o comando:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Opção 2: Usando Anaconda Prompt**
Se você usa Anaconda Prompt, você pode instalar o uv diretamente nele:

1. Abra o **Anaconda Prompt** (não precisa ser como administrador)
2. Execute o comando:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Importante:** Após a instalação no Windows:
- Feche e reabra o Anaconda Prompt (ou PowerShell) para que o `uv` fique disponível
- Para verificar se a instalação funcionou, execute: `uv --version`

**Nota para usuários do Anaconda:** O uv funciona normalmente no Anaconda Prompt. Você pode usar o uv para criar e gerenciar ambientes virtuais mesmo tendo o Anaconda instalado. O uv criará um ambiente virtual separado (na pasta `.venv`) que não interfere com seus ambientes conda.

### Criando o Ambiente Virtual

1. **Sincronizar dependências e criar ambiente:**
   No anaconda prompt, navegue até a pasta do projeto e execute o comando:

   ```bash
   uv sync
   ```
   
   Este comando irá:
   - Criar um ambiente virtual automaticamente (se não existir)
   - Instalar todas as dependências listadas no `pyproject.toml`
   - Criar um arquivo `uv.lock` com versões exatas das dependências

2. **Ativar o ambiente virtual:**

   **No Anaconda Prompt (Windows):**

   No anaconda prompt, navegue até a pasta do projeto e execute o comando:

   ```bash
   .venv\Scripts\activate
   ```

   **Nota:** Com o uv, você também pode executar comandos diretamente sem ativar o ambiente:
   ```bash
   uv run python src/flow.py run
   ```
   
   **Dica para iniciantes:** Se você estiver usando o Anaconda Prompt, após executar `uv sync`, você verá uma pasta `.venv` criada no seu projeto. Para ativar esse ambiente, use o comando acima. Quando o ambiente estiver ativo, você verá `(.venv)` no início da linha do prompt.

### Executando o Flow

Para executar o pipeline Metaflow:

```bash
uv run -m src.flow run
```

Para executar com um número específico de trials no HPO:
```bash
uv run -m src.flow run --n_trials 20
```

### Estrutura de Dados Esperada

Certifique-se de que o arquivo `data/train.csv` existe e contém os dados de treino com as colunas necessárias.

## 5. Exercícios

### Exercício 1: Configuração do Ambiente

**Objetivo:** Configurar o ambiente de desenvolvimento e executar o pipeline completo.

**Tarefas:**
1. Abra o **Anaconda Prompt**
2. Instale o `uv` seguindo as instruções da seção de instalação acima
3. Navegue até a pasta do projeto:
   ```bash
   cd caminho/para/bike-sharing-demand-forecasting
   ```
4. Execute `uv sync` para criar o ambiente virtual e instalar todas as dependências:
   ```bash
   uv sync
   ```
5. Ative o ambiente virtual:
   ```bash
   .venv\Scripts\activate
   ```
   Você deve ver `(.venv)` aparecer no início da linha do prompt
6. Verifique se o arquivo `data/train.csv` existe
7. Execute o flow completo:
   ```bash
   uv run -m src.flow run
   ```
8. Verifique se os arquivos de saída foram criados:
   - `data/processed_train_data.csv`
   - `data/models/baseline_model_*.json`
   - `data/models/hpo/*.json`

**Dica:** Se encontrar erros, verifique se todas as dependências foram instaladas corretamente e se o arquivo de dados está no local correto. Se o comando `uv` não for reconhecido após a instalação, feche e reabra o Anaconda Prompt.

### Exercício 2: Análise dos Resultados

**Objetivo:** Analisar e validar os resultados dos modelos baseline e HPO.

**Tarefas:**
1. Abra o arquivo `data/models/baseline_model_*.json` e verifique o valor do MAPE
2. Abra os arquivos em `data/models/hpo/` e verifique os valores de MAPE de cada modelo otimizado
3. Compare os valores de MAPE:
   - O MAPE do baseline está dentro de uma faixa razoável? (considere que MAPE é uma métrica de erro percentual)
   - Os modelos com HPO melhoraram em relação ao baseline?
   - Qual modelo apresentou o melhor desempenho?
4. Verifique se os valores de MAPE fazem sentido para um problema de previsão de demanda:
   - Valores muito altos (>100%) indicam problemas
   - Valores muito baixos (<1%) podem indicar overfitting ou problemas na métrica
   - Valores entre 10-30% são geralmente aceitáveis para problemas de demanda

**Dica:** Lembre-se que MAPE (Mean Absolute Percentage Error) mede o erro percentual médio. Um MAPE de 20% significa que, em média, o modelo erra 20% em relação ao valor real.

### Exercício 3: Correção do MAPE Explodindo

**Objetivo:** Identificar e corrigir o problema que está causando valores de MAPE muito altos ou infinitos.

**Tarefas:**
1. Execute o flow e observe os valores de MAPE nos arquivos de saída
2. Se o MAPE estiver muito alto (>100%) ou infinito, investigue a causa
3. Analise o código de avaliação em `src/train/baseline_model.py` e `src/hpo/optuna_hpo.py`
4. Verifique a função `mean_absolute_percentage_error` do scikit-learn
5. Identifique o problema e implemente a correção

**Dica:** O problema pode estar relacionado a valores zero ou muito próximos de zero na variável target. A fórmula do MAPE tradicional tem problemas quando há valores zero ou muito pequenos no denominador. Verifique a documentação da função `mean_absolute_percentage_error` do scikit-learn para entender como ela lida com esse caso e considere alternativas como usar uma métrica diferente ou tratar valores zero antes do cálculo.

## 6. Objetivo Final: API FastAPI

O objetivo final deste projeto é construir uma **API REST usando FastAPI** que servirá o melhor modelo treinado para fazer previsões em tempo real. A API permitirá:

- Receber dados de entrada via requisições HTTP
- Fazer previsões usando o melhor modelo selecionado pelo pipeline
- Retornar as previsões de demanda de bicicletas compartilhadas

**Nota:** A construção da API será feita em um segundo momento, após a validação e seleção do melhor modelo através do pipeline atual. O diretório `src/api/` já está preparado para receber a implementação futura da API.

### Próximos Passos (Futuro):
1. Serializar e salvar o melhor modelo treinado
2. Criar endpoints FastAPI para:
   - Health check
   - Previsão de demanda (POST /predict)
   - Informações do modelo (GET /model/info)
3. Implementar validação de dados de entrada
4. Adicionar logging e monitoramento
5. Containerizar a aplicação (Docker)
