# atividade-ViT
Comparação entre os modelos ViT Base e ViT Large para classificação de imagens por URL, com geração automática de resultados e relatório acadêmico em DOCX.

# Relatório ViT — Comparação entre ViT Base e ViT Large

Este projeto foi desenvolvido em Python com o objetivo de realizar testes de classificação de imagens utilizando modelos Vision Transformer, comparando o desempenho entre o `google/vit-base-patch16-224` e o `google/vit-large-patch16-224`. A proposta é executar os modelos sobre imagens obtidas por URL, registrar os principais resultados da inferência e gerar automaticamente um relatório acadêmico em formato `.docx`.

O projeto permite analisar, para cada imagem, a classe principal prevista por cada modelo, o nível de confiança, o Top 5 de probabilidades, o tempo de carregamento e o tempo de inferência. Com isso, é possível observar diferenças entre os modelos em termos de precisão, interpretação da imagem e custo computacional. Em geral, o ViT Base tende a apresentar menor tempo de processamento, enquanto o ViT Large pode exigir mais memória e maior tempo de execução.

## Tecnologias utilizadas

O projeto utiliza Python e bibliotecas voltadas para processamento de imagens, aprendizado profundo, manipulação de dados e geração de documentos. Entre as principais dependências estão `torch`, `torchvision`, `transformers`, `pillow`, `requests`, `pandas`, `matplotlib`, `python-docx` e `tqdm`.

As versões das bibliotecas foram fixadas no arquivo `requirements.txt` para manter maior compatibilidade com Python 3.8 e evitar conflitos durante a execução.

## Estrutura do projeto

```text
vit_relatorio/

├── src/
│   ├── run_vit_tests.py
│   ├── generate_report.py
│   └── utils.py
├── output/
│   ├── imagens/
│   ├── resultados/
│   └── logs/
├── requirements.txt
└── README.md
```

A pasta `src/` concentra os scripts principais do projeto. O arquivo `run_vit_tests.py` é responsável por baixar as imagens, executar os modelos ViT Base e ViT Large e salvar os resultados. O arquivo `generate_report.py` gera o relatório final em `.docx`, enquanto `utils.py` reúne funções auxiliares utilizadas durante o processo.

A pasta `output/` armazena os arquivos produzidos pela execução. As imagens baixadas são salvas em `output/imagens/`, os resultados em `.csv` e `.json` ficam em `output/resultados/`, e os registros completos da execução são armazenados em `output/logs/`.

## Instalação

Para executar o projeto, recomenda-se criar um ambiente virtual antes de instalar as dependências.

No Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução dos testes

Após instalar as dependências, execute o script principal de testes:

```bash
python src/run_vit_tests.py
```

Esse script realiza o download das imagens configuradas, executa os dois modelos Vision Transformer, calcula os resultados da classificação e salva os dados obtidos em arquivos estruturados. Os resultados finais são armazenados em:

```text
output/resultados/resultados_vit.csv
output/resultados/resultados_vit.json
```

Além disso, o log completo da execução é salvo em:

```text
output/logs/execucao.log
```

## Geração do relatório

Depois de executar os testes, o relatório acadêmico pode ser gerado com o comando:

```bash
python src/generate_report.py
```

O documento final será criado em:

```text
output/relatorio_vit_abnt.docx
```

O relatório apresenta os resultados obtidos nos testes, incluindo os comentários comparativos entre os modelos ViT Base e ViT Large, com foco na interpretação das imagens, no tempo de inferência e na diferença entre as previsões geradas.

## Exemplo de execução completa

No Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/run_vit_tests.py
python src/generate_report.py
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_vit_tests.py
python src/generate_report.py
```

## Resultados esperados

Ao final da execução, o projeto deve gerar arquivos com os resultados das classificações realizadas pelos modelos. Esses arquivos permitem comparar o desempenho entre o ViT Base e o ViT Large, observando diferenças na classe prevista, na confiança das previsões e no tempo necessário para processar cada imagem.

O projeto também gera automaticamente um relatório em formato `.docx`, facilitando a organização dos resultados em um documento acadêmico.

## Observações

O modelo `google/vit-large-patch16-224` pode exigir mais memória e apresentar maior tempo de carregamento e inferência em comparação com o modelo Base. Caso ocorram erros relacionados à falta de memória, recomenda-se verificar o arquivo de log em `output/logs/execucao.log` e considerar a execução em uma máquina com GPU ou em ambiente como o Google Colab.

Este projeto foi desenvolvido com finalidade acadêmica, buscando demonstrar a aplicação prática de modelos Vision Transformer na classificação de imagens e a comparação entre arquiteturas de tamanhos diferentes.
