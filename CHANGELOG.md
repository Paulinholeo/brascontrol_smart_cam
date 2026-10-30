# Changelog

Versionamento do Brascontrol Smart Cam (fork de [roryclear/clearcam](https://github.com/roryclear/clearcam), GPL-3.0).
Segue [SemVer](https://semver.org/lang/pt-BR/): `MAJOR.MINOR.PATCH`. A versão atual fica no arquivo `VERSION`, e cada release recebe uma tag `vX.Y.Z`.
As tags `1.x.y` sem `v` são do projeto original.

## [Não lançado]

### Interface
- Animações seguem a skill `emil-design-eng`:
  - transições com propriedades explícitas em vez de `all`;
  - curvas `--ease-out` e `--ease-in-out`;
  - botões reagem ao clique com `scale(0.97)`;
  - efeitos de movimento no hover só com mouse;
  - modais entram com fade e escala;
  - suporte a `prefers-reduced-motion`.
- Celular (skill `mobile-native`):
  - sem flash ao tocar e sem atraso de toque;
  - campos com 16px em tela de toque, para o iPhone não dar zoom;
  - altura com `100dvh`;
  - `theme-color` acompanha o tema claro/escuro.
- Galeria de placas com dados extremos (skill `break-ui`):
  - nome de câmera longo não faz mais a página rolar para o lado;
  - o nome do motor OCR cabe no card;
  - imagem apagada mostra "Imagem indisponível";
  - contadores com separador de milhar;
  - título do modal para leitura sem texto.
- `test/ui_worst_case.js`: dados de pior caso para testar a galeria no navegador.
- `.claude/skills`: skills de design de [emilkowalski/skills](https://github.com/emilkowalski/skills) (MIT).

## [0.2.0] - 2026-10-29

### Desempenho
- As câmeras ao vivo são detectadas numa única chamada em lote na GPU (`YOLOv9.detect_batch`), agrupadas pela resolução após o letterbox, e não mais uma por vez. O VOD continua no caminho antigo.
- O letterbox é feito com OpenCV na CPU. O caminho antigo interpolava em uint8 dentro do tinygrad, o que corrompia cerca de 0,35% dos pixels de cada quadro.
- O quadro não faz mais a ida e volta GPU→CPU a cada inferência.
- `sys.setswitchinterval` baixou de 5 ms para 0,5 ms (variável `CLEARCAM_GIL_SWITCH`). O tinygrad solta o GIL a cada chamada OpenCL, e com cerca de 60 threads a inferência esperava até 5 ms por kernel. A chamada de detecção caiu de ~230 ms para ~16 ms por laço.
- O leitor MJPEG da Pumatronix lia em blocos de 8 KB e reprocessava o buffer inteiro a cada bloco. Agora lê o que estiver disponível e retoma a busca de onde parou.
- A tabela `links` não é mais lida do banco a cada quadro: o laço principal mantém um cache atualizado a cada 5 s.
- `CLEARCAM_PROFILE=1` imprime, a cada 10 s, o tempo de cada etapa do laço principal e o FPS de captura de cada câmera.

Resultado na máquina de desenvolvimento (RTX 3050, 7 câmeras, YOLOv9-t a 640):
- Inferência: de ~1,2 FPS por câmera para o mesmo FPS que a câmera entrega (2–6 FPS).
- Uso da GPU: de 11% para 23%.

O limite agora é a própria Pumatronix. Medida sozinha, a câmera .21 em 1280x720 entrega 4,3 FPS no MJPEG com Q80, 7,7 FPS com Q50 e 10,5 FPS com Q30. Com 4 câmeras ao mesmo tempo, o Q80 cai para cerca de 2,5 FPS.

### Corrigido
- No modo ao vivo, o recorte enviado ao OCR vinha do quadro anterior. Agora vem do mesmo quadro em que o veículo foi detectado.

## [0.1.0] - 2026-10-29

Linha de base congelada antes das correções de perda de placa. Os resultados de contagem e de leitura de placa medidos nesta versão servem de referência para as próximas.

### Adicionado
- Protocolos de câmera: Pumatronix ITSCAM, ONVIF, Hikvision ISAPI/ISUP e descoberta na rede.
- OCR de placas (RapidOCR, Tesseract, EasyOCR, ensemble) com normalização Mercosul.
- Atributos de veículo (cor, tipo), galeria de placas e busca de veículos.
- Identidade visual Brascontrol e scripts de inicialização para Windows.
- `concepts_brief.md`: análise e plano de medição de perdas de LPR e contador de veículos.

### Problemas conhecidos (a corrigir nas próximas versões)
- O OCR roda no máximo uma vez a cada 2 s por câmera, e não por veículo.
- No modo ao vivo, o recorte enviado ao OCR vem do quadro anterior.
- O primeiro recorte de cada veículo só é salvo depois de cerca de 1 s de rastreamento.
- A detecção ao vivo só roda enquanto algum alerta está ativo.
- O servidor HTTP não tem autenticação e `/delete_camera` aceita caminhos relativos (`..`). Não exponha o servidor fora da LAN.
