# Presentación del trabajo

La presentación está implementada con `Markdown`, usando la herramienta [Marp](https://marp.app/). Para poder generar la visualización de la presentación, hay que descargar la CLI de la herramienta anteriormente nombrada:

```sh
npm i -g @marp-team/marp-cli
```

A continuación, se puede generar la visualización en formato HTML:

```sh
marp slide-deck.md
```

o en formado PDF:

```sh
marp --pdf --allow-local-files slides.md
```
