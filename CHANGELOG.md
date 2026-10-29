# Changelog

Versionamento do Brascontrol Smart Cam (fork de [roryclear/clearcam](https://github.com/roryclear/clearcam), GPL-3.0).
Segue [SemVer](https://semver.org/lang/pt-BR/): `MAJOR.MINOR.PATCH`. A versão atual fica no arquivo `VERSION`, e cada release recebe uma tag `vX.Y.Z`.
As tags `1.x.y` sem `v` são do projeto original.

## [Não lançado]

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
