# Gerador de legenda SRT

Transcreve audio/video com Whisper, traduz e gera arquivos .srt.

## Arquivo principal
- gerar_srt.py

## Uso rapido
Na pasta do projeto:

```powershell
python .\gerar_srt.py
```

Sem argumentos, o script abre uma janela para selecionar um ou varios arquivos.
Para cada arquivo selecionado, o .srt e criado na mesma pasta do video/audio.
Antes da traducao, o script gera primeiro um arquivo original com sufixo _sem_traducao.
Depois gera o arquivo final traduzido com o nome normal.
Cada legenda tem no maximo oito palavras; segmentos maiores sao divididos em novas legendas.
Ao iniciar, o script pergunta no terminal:
- 1 = com CUDA (GPU NVIDIA)
- 2 = sem CUDA (CPU)
Depois pergunta o idioma de origem do audio:
- Enter = ingles (en)
- 1 = japones (ja)
- 2 = espanhol (es)
- 3 = portugues (pt)
O terminal mostra:
- painel colorido com configuracao atual
- tabela dos arquivos selecionados
- duracao de cada arquivo
- tempo previsto por arquivo
- barras de progresso por fase e traducao de segmentos
- progresso real em frames durante a transcricao do SRT original
- provedor ativo e quantidade de segmentos traduzidos atualizados a cada lote
- traducao em lotes reais para reduzir a quantidade de requisicoes
- timeout e retentativas limitadas para evitar espera indefinida
- traducao premium em portugues brasileiro pelo DeepL Cloud
- lotes nativos do DeepL preservam cada fala separadamente

## Traducao premium
O script usa exclusivamente o DeepL Cloud para a traducao final. Crie uma chave de API no DeepL e preencha o arquivo `config.json`:

```json
{
	"deepl_api_key": "sua-chave"
}
```

O `config.json` faz parte do repositorio e sera incluido nos commits e pushes. Como solicitado, a chave salva nele ficara visivel no historico Git.
Tambem e possivel usar a variavel de ambiente:

```powershell
$env:DEEPL_API_KEY = "sua-chave"
python .\gerar_srt.py
```

Se a chave nao estiver no `config.json` nem em `DEEPL_API_KEY`, o terminal a solicita com entrada oculta.
Contas DeepL API Free e DeepL API Pro sao detectadas automaticamente.

Tambem e possivel executar por duplo clique no arquivo executar_para_selecionar_videos.bat.

## Uso por linha de comando
Um arquivo:

```powershell
python .\gerar_srt.py .\Donations2.mp4
```

Multiplos arquivos:

```powershell
python .\gerar_srt.py .\video1.mp4 .\video2.mp4
```

Via .bat (com argumentos):

```powershell
.\executar_para_selecionar_videos.bat .\video1.mp4 .\video2.mp4
```

Com opcoes:

```powershell
python .\gerar_srt.py .\Donations2.mp4 --model small --source en --target pt --output legenda_final
```

Definir dispositivo sem pergunta interativa:

```powershell
python .\gerar_srt.py .\Donations2.mp4 --device cuda
python .\gerar_srt.py .\Donations2.mp4 --device cpu
```

Desativar menu de idioma e usar o valor de --source:

```powershell
python .\gerar_srt.py .\Donations2.mp4 --source pt --source-menu off
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
- requests (cliente da API DeepL Cloud)
- imageio-ffmpeg (usado para disponibilizar FFmpeg automaticamente quando nao existir no sistema)
- rich (interface colorida no terminal)

## GPU
Se houver uma GPU NVIDIA disponivel, o script tenta usar CUDA automaticamente.
Se o PyTorch instalado for CPU-only, ele tenta instalar automaticamente uma versao CUDA (cu124 e cu121).
Se nao for possivel (sem internet, permissao, ou incompatibilidade), o script avisa e continua em CPU.

## Observacao
O script verifica FFmpeg no inicio. Se nao encontrar, ele instala automaticamente uma versao via imageio-ffmpeg e usa na execucao atual.
