/* Native checks: dumps logits and token ids for comparison with ref.py / HF tokenizers. */
#define main nicu_main
#include "nicu.c"
#undef main
int main(int argc, char **argv) {
    static Model m; static Tok k; int ids[1024], n, i, toks[] = {5, 99, 1234, 42, 7, 3000, 11};
    model_load(&m, "build/nicu.bin"); tok_load(&k, "build/nicutok.bin");
    for (i = 0; i < 7; i++) forward(&m, toks[i], i);
    for (i = 0; i < 5; i++) printf("%.5f ", m.logits[i]); printf("\n");
    n = tok_encode(&k, argv[1], ids);
    for (i = 0; i < n; i++) printf("%d ", ids[i]); printf("\n");
    return 0;
}
