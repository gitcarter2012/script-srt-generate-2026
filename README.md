# Gerador de legenda SRT

Transcreve audio/video com Whisper, traduz e gera arquivos .srt.

## Arquivo principal
- gerar_srt.py

## Uso rapido
Na pasta do projeto:

```powershell
py .\gerar_srt.py
```

Sem argumentos, o script abre uma janela para selecionar um ou varios arquivos.
Para cada arquivo selecionado, o .srt e criado na mesma pasta do video/audio.
O terminal mostra:
- painel colorido com configuracao atual
- tabela dos arquivos selecionados
- duracao de cada arquivo
- tempo previsto por arquivo
- barras de progresso por fase e traducao de segmentos

Tambem e possivel executar por duplo clique no arquivo executar_para_selecionar_videos.bat.

## Uso por linha de comando
Um arquivo:

```powershell
py .\gerar_srt.py .\Donations2.mp4
```

Multiplos arquivos:

```powershell
py .\gerar_srt.py .\video1.mp4 .\video2.mp4
```

Via .bat (com argumentos):

```powershell
.\executar_para_selecionar_videos.bat .\video1.mp4 .\video2.mp4
```

Com opcoes:

```powershell
py .\gerar_srt.py .\Donations2.mp4 --model small --source en --target pt --output legenda_final
```

Observacao: use --output apenas com um unico arquivo de entrada.

## Parametros
- input: caminho(s) do(s) arquivo(s).
- --model: tiny, base, small, medium, large.
- --source: idioma do audio (padrao: en).
- --target: idioma de traducao (padrao: pt).
- --output: nome base do arquivo de saida (sem extensao).

## Dependencias
O script verifica e instala automaticamente (via pip), quando necessario:
- openai-whisper
- deep-translator
- imageio-ffmpeg (usado para disponibilizar FFmpeg automaticamente quando nao existir no sistema)
- rich (interface colorida no terminal)

## GPU
Se houver uma GPU NVIDIA disponivel, o script tenta usar CUDA automaticamente.
Se o PyTorch instalado for CPU-only, ele tenta instalar automaticamente uma versao CUDA (cu124 e cu121).
Se nao for possivel (sem internet, permissao, ou incompatibilidade), o script avisa e continua em CPU.

## Observacao
O script verifica FFmpeg no inicio. Se nao encontrar, ele instala automaticamente uma versao via imageio-ffmpeg e usa na execucao atual.
