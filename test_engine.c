/* Native check: prints the first logits after a fixed token sequence and the token ids of a text,
   for comparison with ref.py and Hugging Face tokenizers. usage: test_engine DAT_DIR "text" */
#define main nicu_main
#include "nicu.c"
#undef main
int main(int argc, char **argv) {
    static Model m; static Tok k; static char path[1024];
    int ids[MAXIDS], n, i, toks[] = {5, 99, 1234, 42, 7, 3000, 11};
    if (argc < 3) { fprintf(stderr, "usage: %s DAT_DIR \"text\"\n", argv[0]); return 2; }
    sprintf(path, "%.1000s/nicu.dat", argv[1]);
    if (!model_load(&m, path)) { fprintf(stderr, "cannot load %s\n", path); return 1; }
    sprintf(path, "%.1000s/nicutok.dat", argv[1]);
    if (!tok_load(&k, path)) { fprintf(stderr, "cannot load %s\n", path); return 1; }
    for (i = 0; i < 7; i++) forward(&m, toks[i], i);
    for (i = 0; i < 5; i++) printf("%.5f ", m.logits[i]); printf("\n");
    n = tok_encode(&k, argv[2], ids, MAXIDS);
    for (i = 0; i < n; i++) printf("%d ", ids[i]); printf("\n");
    return 0;
}
