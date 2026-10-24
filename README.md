# Nicu per Mac OS 9

Motore di inferenza in C che fa girare [Nicu](https://huggingface.co/federico-anastasi/nicu-20m) su un Power Mac G4 con Mac OS 9.1. Con Nicu 20M a 8 bit e AltiVec arriva a circa 5,6 token al secondo.

## File

- `nicu.c`: motore (GPT tipo nanoGPT, tokenizer BPE a byte, campionamento top-p con penalità di ripetizione) e chat. Si compila anche sul PC per i test.
- `export.py`: converte `checkpoint.pt` e `tokenizer.json` di Hugging Face in `nicu.dat` (pesi) e `nicutok.dat` (vocabolario), big-endian.
- `ref.py`: forward in numpy sui pesi esportati, confrontato con l'ONNX fp32 ufficiale.
- `test_engine.c`: stampa logit e token del motore C, da confrontare con `ref.py` e con `tokenizers`.
- `console/`: copia della console di Retro68 con font Monaco 18 e titolo della finestra (l'originale ha Monaco 9 fisso e ignora il titolo).
- `nicu.r`: memoria dell'app (64 MB preferiti, 40 minimi).
- `.github/workflows/build.yml`: compila l'app PowerPC con l'immagine Docker di Retro68 e pubblica `Nicu.bin` (MacBinary) come artifact.

## Formato di `nicu.dat` (versione 3)

Intestazione `NICU` + 7 interi big-endian: versione, vocabolario, contesto, layer, teste, dimensione, pesi legati. Poi i tensori nell'ordine del modello. Le matrici grandi (embedding, qkv, proiezioni, MLP) sono int8 con una scala float per riga, e le righe int8 partono allineate a 16 byte. Tutto il resto è float32. Rispetto all'ONNX fp32 il token più probabile coincide nel 100% dei casi.

## Comandi

```
python export.py <cartella-hf> ../strumenti/sito          # hf: checkpoint.pt + tokenizer.json (+ model.onnx per ref.py)
python ref.py ../strumenti/sito <cartella-hf>             # confronto con l'ONNX
python -m ziglang cc -O2 -o nicu_test.exe nicu.c          # test sul PC: eseguilo accanto a nicu.dat e nicutok.dat
```

Sul Mac l'app cerca `nicu.dat` e `nicutok.dat` nella sua stessa cartella. Si chiude scrivendo `esci` o con la casella di chiusura della finestra.

## Cose da sapere su Mac OS 9 e Retro68

- **Lo stack non è allineato a 16 byte** (misurato: scarto di 12). Le istruzioni AltiVec ignorano i bit bassi dell'indirizzo, quindi un vettore che passa dallo stack viene letto o scritto nel posto sbagliato. Per questo `matvec_altivec` è `noinline`, prende costanti e somma da globali allineati, e il progetto si compila con `-fno-tree-vectorize -fno-tree-slp-vectorize`: con la vettorizzazione automatica anche il codice normale dava risultati sbagliati.
- All'avvio `altivec_check` confronta AltiVec con il calcolo normale su una matrice vera: se non coincidono, usa il calcolo normale e lo scrive a schermo.
- Retro68 non include `altivec.h`: si usano direttamente le funzioni `__builtin_altivec_*`, con flag `-mcpu=7400 -maltivec -mabi=altivec -mvrsave -Wa,-maltivec`.
- Il multitasking è cooperativo: durante la generazione `yield_cpu()` chiama `WaitNextEvent` dopo ogni token, altrimenti il server VNC e lo schermo si bloccano. Il caricamento dei pesi avviene a blocchi da 1 MB, con un puntino e una pausa per blocco.
- Testo: la console lavora in Mac Roman, il modello in UTF-8. La conversione è in `mac_to_utf8` e `emit_bytes`.
