# Medição de perdas de LPR e contador de veículos (Brascontrol clearcam)

* **Autor:** Paulo Leonardo
* **Data:** 2026-10-29

## Visão geral

O clearcam adaptado pela Brascontrol lê placas (LPR) de câmeras Pumatronix ITSCAM, Hikvision e Intelbras. No local de teste (3 faixas, sentido único), passam cerca de 10 veículos e o sistema registra a imagem de cerca de 2. Hoje não há contador de veículos nem forma de medir quantos veículos são perdidos.

"Perder um veículo" tem dois significados, que precisam ser medidos separadamente:

* **(a) Placa não lida:** o veículo foi detectado, mas não há leitura de placa.
* **(b) Passagem não registrada:** o sistema nem percebeu que o veículo passou.

Uso: interno, pela equipe Brascontrol, na LAN, na máquina de desenvolvimento. O Smart Sampa entra apenas como fonte de ideias. DDNS, VPS e autenticação ficam para a próxima fase.

## Análise

Seis revisores independentes analisaram a hipótese. Conferi no código os pontos em que eles divergiram ou em que a conclusão muda o plano.

### A ordem estava errada: medir antes de corrigir

Os seis revisores chegaram a isso de forma independente. Se os ajustes entrarem antes de existir uma linha de base, ninguém consegue dizer qual deles mudou o "2 de 10". Cada passo deve mudar uma variável e ser medido de novo.

### A causa mais provável não estava na hipótese original

O veículo precisa ficar tempo demais na imagem para ser salvo:

* **Rastreamento:** o OC-SORT só exibe um track depois de 3 detecções seguidas (`min_hits=3`, `ocsort_tracker/ocsort.py:165`).
* **Primeiro recorte:** quando um track novo aparece, `pred_occs[id] = [agora]` e o recorte só é salvo quando `agora - último > 1 s` (`clearcam.py:853-856`). Ou seja, o primeiro recorte só sai depois de cerca de 1 s.
* **Um único laço para tudo:** uma só thread roda a inferência de todas as câmeras, mais a fila de tarefas, o CLIP, as faces e a gravação de arquivos (`clearcam.py:368-372`). Quadros que chegam enquanto ela está ocupada são sobrescritos sem contagem.

Um veículo que fica cerca de 1 s na imagem, com 3 a 4 quadros processados por segundo, nunca é salvo e nunca passa pelo OCR.

### Outras causas confirmadas

* **OCR limitado por câmera, não por veículo:** no máximo uma leitura a cada 2 s por câmera (`clearcam.py:582-585`). É a causa principal de (a).
* **Recorte do quadro anterior:** o recorte vem do quadro anterior no modo ao vivo (`clearcam.py:571` lê `last_frames[-1]`; o quadro atual só entra em `:719`). No modo VOD a ordem é a inversa (`:698`). Por isso **validar só com arquivo de vídeo esconde esse bug**.
* **Recorte em dobro:** o recorte tem o dobro do tamanho da caixa do veículo (`clearcam.py:565-566`). Ele pega as faixas vizinhas, e o OCR pode devolver a placa de **outro** veículo.
* **Normalizador generoso demais:** o normalizador de placa (`models/ocr.py:36-54`) percorre qualquer texto com uma janela de 7 caracteres e troca 0→O, S→5 etc. Adesivos e telefones podem virar placas "válidas" e contar como acerto.
* **Falhas sem registro:** leituras que falham não deixam rastro. `ocr_events` só é gravado quando há resultado (`clearcam.py:2754`), e a deduplicação por texto em 5 s pode esconder um segundo veículo.

### A detecção pode estar desligada sem ninguém saber

O modo ao vivo só roda a inferência quando algum alerta está ativo no horário (`clearcam.py:715-717`). Se alguém apagar ou reagendar o alerta padrão, param a detecção, o OCR e qualquer contador novo, sem nenhum aviso. O modo VOD não tem esse filtro, então os dois modos se comportam de forma diferente.

### IDs de track não são únicos

`KalmanBoxTracker.count` é zerado a cada `OCSort.__init__` (`ocsort_tracker/ocsort.py:179`). Isso acontece para todas as câmeras quando uma câmera é adicionada e no reinício diário. Como os IDs reaproveitados continuam em `object_set` e `pred_occs`, que nunca são limpos, esses veículos são ignorados.

### Correção de um item do plano anterior

A suposta corrida na troca de modelo (`clearcam.py:2811`) não existe. `set_settings` passa por `add_to_queue` (`clearcam.py:2170`) e roda na mesma thread da inferência. O problema real é outro: recarregar o modelo ou o Qwen **trava todas as câmeras** durante o download e a compilação, sem nenhum log.

### Limiares descartam detecções noturnas

O detector entrega caixas a partir de 0.25, mas o tracker descarta tudo abaixo do limiar da zona (padrão 0.5, `clearcam.py:848`). O modo BYTE (`use_byte=False`), que reaproveitaria essas caixas fracas, está desligado. À noite e com infravermelho, essa perda deve aumentar.

### A medição precisa ser repetível

Os controles de tempo usam o relógio de parede (`time.time()`), então o resultado depende da velocidade da máquina. Além disso, `process_frame_vod` lê um quadro e descarta o seguinte (`grab()` + `read()`). São necessários dois níveis de teste:

1. **Reprocessamento determinístico do arquivo:** todos os quadros, com o tempo calculado como `quadro / fps`. Duas execuções do mesmo vídeo devem dar resultados idênticos.
2. **Reprodução como ao vivo:** o vídeo é retransmitido por RTSP (por exemplo, mediamtx + `ffmpeg -re`) pelo caminho real, para medir desempenho e quadros perdidos.

### Pergunta de produto ainda aberta

O projeto já tem `PumatronixClient.trigger_snapshot` (`protocols/pumatronix.py:139`), e a ITSCAM tem trigger e OCR embarcados. Antes de investir em LPR por software, falta decidir:

* **(a)** o clearcam é um LPR por software para câmeras sem LPR próprio, ou
* **(b)** é um agregador e galeria das placas que as câmeras já leem?

Qualquer que seja a resposta, a leitura da própria câmera no mesmo vídeo serve como uma segunda referência, sem custo.

## Objetivos e não objetivos

* **Objetivos desta fase:**
  * Linha de base medida com o código atual, num vídeo de dia e, se possível, num de noite.
  * Arquivo de referência (contagem manual) por vídeo.
  * Contadores de descarte com motivo, por câmera e por minuto.
  * Correção do recorte com o quadro errado e do intervalo de 1 s antes de salvar.
  * OCR por veículo (vários candidatos por track, com votação) no lugar do limite de 2 s por câmera.
  * Detecção sempre ligada, independente dos alertas.
  * Módulo puro de contagem por linha de cruzamento.
  * Tabela `passages` tipada, com o motivo de cada perda.
* **Não objetivos desta fase:**
  * DDNS e exposição na internet (o servidor não tem login e tem um `rmtree` com path traversal em `/delete_camera`).
  * VPS.
  * Autenticação.
  * Reconhecimento facial.
  * Troca do detector (ONNX).
  * Troca do tracker.
  * Reescrita do `clearcam.py`.
  * Migração dos settings em pickle.

## Alternativas avaliadas

* **Contar só por ID de track:** rejeitada. Uma troca de ID (caminhão na frente) conta o veículo duas vezes, e veículos parados também são contados.
* **Linha sem tracker:** rejeitada. Sem tracker não há como saber se é o mesmo veículo em quadros seguidos.
* **Uma linha por faixa (proposta original):** substituída. Um veículo que troca de faixa cruza duas linhas. Três revisores recomendam uma única linha atravessando a pista, com a faixa definida pela posição x do ponto de cruzamento.
* **Corrigir os bugs primeiro e medir depois:** rejeitada. Sem linha de base, não dá para atribuir a melhora a nenhuma correção.
* **Validar só com arquivo de vídeo (VOD):** rejeitada sozinha. O VOD esconde o bug do quadro anterior e o filtro de alertas.
* **Trocar o detector por um modelo mais novo agora:** adiada. Nada mostra que a detecção é o gargalo. Um detector de placa em resolução cheia provavelmente rende mais que trocar o detector de veículos.
* **Editor de linhas na interface já nesta fase:** adiado. Para o diagnóstico bastam as linhas num JSON e um script offline. A interface vem depois, reaproveitando o editor de zona (`mainview.html:2458-2487`).
* **`supervision.LineZone`:** possível, mas tem o mesmo problema com troca de ID. O teste de cruzamento são cerca de 20 linhas; o trabalho de verdade está na histerese e na deduplicação.
* **Estender o banco em pickle para as passagens:** rejeitada. `replace=False` faz `MAX(CAST(id))` a cada inserção, e `run_get(None)` desserializa a tabela inteira.

## Decisões tomadas e preparação para os próximos passos

1. **Ordem de trabalho:** cada passo muda uma variável e é medido de novo.
   0. Commit ou tag da linha de base. Backup de `data/cc_cache.db` com os arquivos `-wal` e `-shm`. Gravação do vídeo e contagem manual.
   1. Medição com o código atual.
   2. Correção do recorte com o quadro errado e do intervalo de 1 s; nova medição.
   3. OCR por veículo, com recorte 1x e margem; nova medição.
   4. Contador ligado ao pipeline ao vivo e à tabela de passagens.
   5. Higiene: Qwen fora da thread de inferência, `use_notifs` (`:758`), cache de `links`, recarga do modelo.
   6. Detector, adiado.
2. **Definição de passagem:** um veículo físico cruzando a linha da pista uma vez, no sentido configurado. Detalhes:
   * O ponto de referência é a base central da caixa.
   * Há histerese: o veículo precisa avançar uma distância mínima depois da linha.
   * Uma segunda contagem é suprimida quando um ID novo aparece perto de onde um ID antigo acabou de sumir.
   * A posição é interpolada entre quadros, para que um salto rápido não pule a linha.
   * O cruzamento é testado sobre todos os tracks, antes dos filtros de classe, velocidade e zona.
3. **Chave da passagem:** (câmera, sessão do tracker, track_id, linha), porque os IDs de track se repetem.
4. **Métricas, com limites definidos antes do teste:**
   * acerto da contagem por faixa (meta: erro ≤ 5%)
   * taxa de leitura, só entre placas legíveis para um humano
   * acerto exato da placa
   * A detecção que falha (perda b) só pode ser medida contra a contagem manual. O painel ao vivo mostra apenas a taxa de leitura entre os veículos detectados (perda a).
5. **Três níveis de "placa lida":**
   1. algum texto retornado
   2. formato válido, com uma marca quando houve troca de caracteres
   3. igual à placa da contagem manual
   * Em todos os casos, guardar o texto bruto, a confiança e o motor de OCR.
6. **Armazenamento:**
   * Nova tabela SQLite `passages`, com tipos, indexada por (câmera, faixa, horário).
   * As linhas ficam como dicionários simples nos settings de cada câmera.
   * **Não** alterar `VERSION` em `utils/db.py:9`, que entra no nome de todas as tabelas.
   * Editar só as linhas não pode disparar o `reset_vod`.
7. **Código:**
   * Módulos novos e puros: `counting/line_counter.py` e `storage/passages.py`.
   * Chamados de um único ponto, logo após `tracker.update` (`clearcam.py:849`).
   * `frame` e o horário do quadro são passados explicitamente para `save_object` e para o contador.
8. **Precauções que custam pouco já nesta fase:**
   * nunca abrir porta no roteador
   * validar o nome da câmera antes do `rmtree` em `:1479`
   * apagar e colocar no `.gitignore` os `init_*.mp4` da raiz
   * desligar Qwen, CLIP e face durante o teste

### Decisões em aberto

1. **Escopo do produto:** LPR por software ou agregador das leituras da câmera?
2. **Geometria da contagem:** uma linha atravessando a pista com faixas por posição (recomendado) ou uma linha por faixa?
3. **Câmera de referência do teste:** qual delas, e se há também um vídeo noturno disponível.
