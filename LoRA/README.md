# Fine-Tuning de Modelos de Linguagem: LoRA, QLoRA, PEFT e Fine-Tuning Completo

Este material apresenta uma explicação didática e técnica sobre diferentes abordagens de fine-tuning (ajuste fino) em modelos de linguagem natural, com foco nas técnicas **LoRA (Low-Rank Adaptation)**, **QLoRA (Quantized LoRA)**, **PEFT (Parameter-Efficient Fine-Tuning)** e **Fine-Tuning Completo**. A proposta é servir como referência para Cientistas de Dados e Engenheiros de IA que desejam compreender, comparar e aplicar essas estratégias em modelos como **BERT**, **LLaMA** e **T5**.

---

## Índice

1. [Introdução](#introdu%C3%A7%C3%A3o)
2. [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
   - Como funciona
   - Arquitetura
   - Vantagens e Desvantagens
   - Implicações Computacionais
   - Aplicações em BERT, LLaMA e T5
3. [QLoRA (Quantized LoRA)](#qlora-quantized-lora)
   - Fundamentos
   - Arquitetura e Implementação
   - Vantagens e Desvantagens
   - Implicações Computacionais
   - Comparações de Eficiência
4. [PEFT (Parameter-Efficient Fine-Tuning)](#peft-parameter-efficient-fine-tuning)
   - Técnicas: Adapters, Prompt/Prefix Tuning, BitFit
   - Alterações de Arquitetura
   - Vantagens e Desvantagens
   - Comparações de Desempenho
5. [Fine-Tuning Completo](#fine-tuning-completo)
   - Características
   - Vantagens e Desvantagens
   - Implicações Computacionais
6. [Comparações Finais e Tabela Resumo](#comparações-finais-e-tabela-resumo)

---

## Introdução

Fine-tuning permite especializar modelos pré-treinados em tarefas específicas. Métodos como LoRA, QLoRA e PEFT oferecem formas eficientes de realizar esse ajuste sem atualizar todos os bilhões de parâmetros dos modelos modernos.

Essas técnicas tornaram-se fundamentais com o crescimento dos LLMs (Large Language Models), pois reduzem drasticamente:

- A quantidade de parâmetros treináveis
- O uso de memória e tempo de treinamento
- O risco de overfitting e *catastrophic forgetting*

---

## LoRA (Low-Rank Adaptation)

**LoRA** injeta matrizes de baixa rank ($A$ e $B$) nas projeções lineares de camadas do Transformer. A atualização dos pesos é feita via:

```
ΔW = B × A
```

A saída da camada passa a ser:

```
W × x + α × B × A × x
```

### Arquitetura

- O modelo base permanece congelado
- Apenas os parâmetros $A$ e $B$ são treináveis
- Pode ser aplicado em camadas de atenção (Q, K, V) ou feedforward
- Ao final do treinamento, os deltas LoRA podem ser fundidos no modelo

### Vantagens

- Reduz drasticamente os parâmetros treináveis (~0.1%)
- Usa menos memória (sem necessidade de gradientes para o modelo base)
- Preserva conhecimento pré-treinado
- Pode ser fundido sem impactar a inferência
- Modular: múltiplas tarefas com mesmo modelo base

### Desvantagens

- Capacidade limitada ao rank escolhido
- Requer escolha cuidadosa de onde aplicar
- Overhead mínimo na engenharia (inserção dos módulos)

### Implicações Computacionais

- Redução de 2x a 10x na memória durante o treino
- Pode acelerar o treinamento
- Nenhum custo adicional na inferência (se fundido)

### Aplicações

- **BERT**: LoRA aplicado nas camadas de atenção do encoder
- **LLaMA**: LoRA aplicado no decoder autoregressivo, usado em modelos como Alpaca e Guanaco
- **T5**: Pode ser inserido em encoder e decoder, útil para tarefas seq2seq

---

## QLoRA (Quantized LoRA)

**QLoRA** combina LoRA com quantização do modelo base para **4 bits (int4)**, permitindo treinar modelos massivos em GPUs comuns.

### Fundamentos

- Modelo base congelado e quantizado
- Aplicação de LoRA sobre pesos quantizados
- Usa quantização NF4 (NormalFloat 4) e double quantization

### Arquitetura

- Igual à do LoRA, exceto que os pesos do modelo são armazenados em int4
- Os parâmetros LoRA continuam em float16

### Vantagens

- Redução drástica de memória (ex: LLaMA-65B cabe em 48GB)
- Desempenho comparável ao LoRA normal e ao fine-tuning completo
- Viabiliza treino de modelos >30B em GPU única

### Desvantagens

- Requer infraestrutura para lidar com quantização
- Ligeiro overhead em tempo por passo (dependendo do hardware)

### Implicações Computacionais

- Redução de memória até ~20x comparado ao fine-tuning completo
- Tempo de treino similar ao LoRA puro
- Inferência pode ser feita com modelo quantizado ou convertido

### Aplicações

- **LLaMA**: QLoRA usado no treinamento do Guanaco-65B
- **T5**: Aplicado em modelos de até 11B com sucesso
- **BERT**: Possível, mas menos vantajoso por ser pequeno

---

## PEFT (Parameter-Efficient Fine-Tuning)

Abrange diversas técnicas além de LoRA:

### Técnicas

- **Adapters**: Inserção de módulos com projeções de baixo rank nas camadas
- **Prompt/Prefix Tuning**: Tokens adicionais aprendidos inseridos no input ou nas atenções
- **BitFit**: Ajuste apenas dos biases

### Arquitetura

- Variam na forma de modificação:
  - Adapters adicionam camadas
  - Prompt Tuning adiciona tokens
  - BitFit apenas libera gradientes nos biases

### Vantagens

- Economia de memória e armazenamento
- Modularidade e reuso
- Melhor controle em multitarefa

### Desvantagens

- Ligeiro overhead na inferência (exceto BitFit)
- Performance levemente inferior ao fine-tuning completo em alguns casos

### Comparações de Desempenho

| Técnica         | Acurácia (GLUE - BERT) | Params Treináveis |
|----------------|--------------------------|--------------------|
| Full Fine-Tune | 80.4                     | 100%               |
| Adapters       | 80.0                     | ~3%                |
| LoRA           | 80.1                     | ~0.1%              |
| Prompt Tuning  | 79.0                     | <0.1%              |

---

## Fine-Tuning Completo

### Características

- Todos os parâmetros do modelo são atualizados
- Maior flexibilidade e capacidade de adaptação

### Vantagens

- Melhor desempenho em tarefas com muitos dados
- Arquitetura original mantida (sem adições)
- Simples de implementar

### Desvantagens

- Altíssimo custo de memória e tempo
- Risco de overfitting e *catastrophic forgetting*
- Impraticável para modelos muito grandes

### Implicações Computacionais

- Requer até 3× a memória do modelo para armazenar gradientes e otimizadores
- Necessita múltiplas GPUs ou paralelismo de modelo
- Inferência é padrão (modelo não muda)

---

## Comparações Finais e Tabela Resumo

| Técnica             | Params Treináveis | Memória Treino | Overhead Inferência | Desempenho Relativo |
|---------------------|-------------------|----------------|----------------------|----------------------|
| Fine-Tuning Completo| 100%              | Altíssima      | Nenhum               | 100% (baseline)      |
| LoRA                | ~0.1%             | Baixa          | Nenhum (se fundido)  | ~100%                |
| QLoRA               | ~0.1%             | Mínima         | Nenhum (ou leve)     | ~100%                |
| Adapters            | ~3–5%             | Baixa          | Leve                 | ~99%                 |
| Prompt Tuning       | <0.1%             | Muito baixa    | Leve (tokens extras) | ~95–100%             |
| BitFit              | <0.1%             | Muito baixa    | Nenhum               | ~92–97%              |

---

## Considerações Finais

Técnicas de fine-tuning eficiente como LoRA, QLoRA e PEFT representam um avanço crucial na IA moderna, possibilitando personalizar LLMs com poucos recursos computacionais. 

- **LoRA** é o padrão atual para ajuste de grandes modelos
- **QLoRA** permite fine-tuning de LLMs gigantes em GPUs comuns
- **Adapters e Prompt Tuning** são úteis em multitarefa e modularidade
- **Fine-tuning completo** ainda tem seu lugar, mas é custoso e, em muitos casos, desnecessário

A escolha da técnica depende do tamanho do modelo, disponibilidade de hardware, e requisitos da tarefa. O domínio dessas abordagens é essencial para qualquer profissional de IA que deseja trabalhar com LLMs em ambientes de produção.

---

> Material baseado em pesquisas de Hu et al. (2021), Dettmers et al. (2023), Houlsby et al. (2019), Lester et al. (2021) e outros. Utilizado em apresentações técnicas para cientistas de dados e engenheiros de IA.
