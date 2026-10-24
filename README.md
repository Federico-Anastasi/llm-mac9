# Nicu per Mac OS 9

Motore di inferenza in C per far girare [Nicu](https://huggingface.co/federico-anastasi/nicu-5m) su un Power Mac G4 con Mac OS 9.

- `export.py` converte checkpoint e tokenizer nei file big-endian `nicu.dat` e `nicutok.dat`.
- `nicu.c` contiene il motore (GPT, tokenizer BPE, chat). Si compila anche sul PC per i test.
- GitHub Actions compila l'app PowerPC con Retro68: il risultato è `Nicu.bin` (MacBinary).

Sul Mac: `Nicu`, `nicu.dat` e `nicutok.dat` nella stessa cartella, poi doppio clic su Nicu.
