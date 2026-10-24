/* Nicu on Mac OS 9: a tiny GPT inference engine + chat loop.
 * Builds natively (for testing) or with Retro68 for PowerPC Mac OS (MAC_TARGET). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef MAC_TARGET
#include <Events.h>
#endif

/* ---------- model ---------- */
typedef struct {
    int V, T, L, H, C;
    long nparams;
    float *wte, *wpe, *lnfw, *lnfb;
    struct Layer { float *ln1w, *ln1b, *qkvw, *qkvb, *ow, *ob, *ln2w, *ln2b, *fcw, *fcb, *pw, *pb; } *layer;
    float *kc, *vc;                       /* KV cache: [L][T][C] */
    float *x, *xb, *qkv, *att, *hb, *logits;
} Model;

static int read_be32(const unsigned char *p) { return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]; }

static void *load_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    void *buf;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    buf = malloc(*size);
    if (buf && fread(buf, 1, *size, f) != (size_t)*size) { free(buf); buf = NULL; }
    fclose(f);
    return buf;
}

static int model_load(Model *m, const char *path) {
    long size, i, n;
    unsigned char *b = load_file(path, &size);
    float *w;
    int l, C;
    union { unsigned int u; unsigned char c[4]; } probe;
    if (!b || memcmp(b, "NICU", 4) != 0) return 0;
    m->V = read_be32(b + 8); m->T = read_be32(b + 12); m->L = read_be32(b + 16);
    m->H = read_be32(b + 20); m->C = read_be32(b + 24);
    C = m->C;
    w = (float *)(b + 32);
    n = (size - 32) / 4;
    m->nparams = n;
    probe.u = 1;
    if (probe.c[0] == 1) {                /* little-endian host: swap the big-endian floats */
        unsigned char *p = (unsigned char *)w, t;
        for (i = 0; i < n; i++, p += 4) { t = p[0]; p[0] = p[3]; p[3] = t; t = p[1]; p[1] = p[2]; p[2] = t; }
    }
    m->wte = w; w += (long)m->V * C;
    m->wpe = w; w += (long)m->T * C;
    m->layer = malloc(sizeof(struct Layer) * m->L);
    for (l = 0; l < m->L; l++) {
        struct Layer *y = &m->layer[l];
        y->ln1w = w; w += C;          y->ln1b = w; w += C;
        y->qkvw = w; w += 3 * C * C;  y->qkvb = w; w += 3 * C;
        y->ow = w;   w += C * C;      y->ob = w;   w += C;
        y->ln2w = w; w += C;          y->ln2b = w; w += C;
        y->fcw = w;  w += 4 * C * C;  y->fcb = w;  w += 4 * C;
        y->pw = w;   w += 4 * C * C;  y->pb = w;   w += C;
    }
    m->lnfw = w; w += C; m->lnfb = w; w += C;
    m->kc = calloc((long)m->L * m->T * C, sizeof(float));
    m->vc = calloc((long)m->L * m->T * C, sizeof(float));
    m->x = malloc(C * sizeof(float)); m->xb = malloc(C * sizeof(float));
    m->qkv = malloc(3 * C * sizeof(float)); m->att = malloc(m->T * sizeof(float));
    m->hb = malloc(4 * C * sizeof(float)); m->logits = malloc(m->V * sizeof(float));
    return m->kc && m->vc && m->logits;
}

static void layernorm(float *o, const float *x, const float *w, const float *b, int n) {
    float mean = 0, var = 0, s;
    int i;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    s = 1.0f / (float)sqrt(var / n + 1e-5f);
    for (i = 0; i < n; i++) o[i] = (x[i] - mean) * s * w[i] + b[i];
}

/* o[r] = W[r,:] . x + b[r]   (W is rows x cols, row-major) */
static void matvec(float *o, const float *W, const float *x, const float *b, int rows, int cols) {
    int r, c;
    for (r = 0; r < rows; r++) {
        const float *w = W + (long)r * cols;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (c = 0; c + 3 < cols; c += 4) { s0 += w[c] * x[c]; s1 += w[c+1] * x[c+1]; s2 += w[c+2] * x[c+2]; s3 += w[c+3] * x[c+3]; }
        for (; c < cols; c++) s0 += w[c] * x[c];
        o[r] = s0 + s1 + s2 + s3 + (b ? b[r] : 0);
    }
}

static float *forward(Model *m, int token, int pos) {
    int C = m->C, H = m->H, hd = C / H, l, h, t, i;
    float scale = 1.0f / (float)sqrt((float)hd);
    for (i = 0; i < C; i++) m->x[i] = m->wte[(long)token * C + i] + m->wpe[(long)pos * C + i];
    for (l = 0; l < m->L; l++) {
        struct Layer *y = &m->layer[l];
        float *kc = m->kc + (long)l * m->T * C, *vc = m->vc + (long)l * m->T * C;
        layernorm(m->xb, m->x, y->ln1w, y->ln1b, C);
        matvec(m->qkv, y->qkvw, m->xb, y->qkvb, 3 * C, C);
        memcpy(kc + (long)pos * C, m->qkv + C, C * sizeof(float));
        memcpy(vc + (long)pos * C, m->qkv + 2 * C, C * sizeof(float));
        for (h = 0; h < H; h++) {
            float *q = m->qkv + h * hd, mx = -1e30f, sum = 0, *out = m->xb + h * hd;
            for (t = 0; t <= pos; t++) {
                float *k = kc + (long)t * C + h * hd, s = 0;
                for (i = 0; i < hd; i++) s += q[i] * k[i];
                s *= scale; m->att[t] = s; if (s > mx) mx = s;
            }
            for (t = 0; t <= pos; t++) { m->att[t] = (float)exp(m->att[t] - mx); sum += m->att[t]; }
            for (i = 0; i < hd; i++) out[i] = 0;
            for (t = 0; t <= pos; t++) {
                float a = m->att[t] / sum, *v = vc + (long)t * C + h * hd;
                for (i = 0; i < hd; i++) out[i] += a * v[i];
            }
        }
        matvec(m->qkv, y->ow, m->xb, y->ob, C, C);          /* qkv reused as scratch */
        for (i = 0; i < C; i++) m->x[i] += m->qkv[i];
        layernorm(m->xb, m->x, y->ln2w, y->ln2b, C);
        matvec(m->hb, y->fcw, m->xb, y->fcb, 4 * C, C);
        for (i = 0; i < 4 * C; i++) m->hb[i] = 0.5f * m->hb[i] * (1.0f + (float)erf(m->hb[i] * 0.70710678f));
        matvec(m->qkv, y->pw, m->hb, y->pb, C, 4 * C);
        for (i = 0; i < C; i++) m->x[i] += m->qkv[i];
    }
    layernorm(m->xb, m->x, m->lnfw, m->lnfb, C);
    matvec(m->logits, m->wte, m->xb, NULL, m->V, C);       /* tied output head */
    return m->logits;
}

/* ---------- tokenizer (byte-level BPE) ---------- */
typedef struct { int a, b, out; } Merge;
typedef struct {
    int V, nm;
    unsigned char **tok; unsigned char *len;
    int byte_id[256];
    Merge *merge;
} Tok;

static int tok_load(Tok *k, const char *path) {
    long size; int i;
    unsigned char *b = load_file(path, &size), *p;
    if (!b || memcmp(b, "NTOK", 4) != 0) return 0;
    k->V = read_be32(b + 4); p = b + 8;
    k->tok = malloc(k->V * sizeof(char *)); k->len = malloc(k->V);
    for (i = 0; i < k->V; i++) { k->len[i] = *p; k->tok[i] = p + 1; p += 1 + *p; }
    for (i = 0; i < 256; i++, p += 4) k->byte_id[i] = read_be32(p);
    k->nm = read_be32(p); p += 4;
    k->merge = malloc(k->nm * sizeof(Merge));
    for (i = 0; i < k->nm; i++, p += 12) { k->merge[i].a = read_be32(p); k->merge[i].b = read_be32(p + 4); k->merge[i].out = read_be32(p + 8); }
    return 1;
}

static int is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int is_space(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

/* BPE-merge one pre-token chunk; appends ids to out, returns count */
static int bpe_chunk(Tok *k, const unsigned char *s, int n, int *out) {
    int ids[512], cnt = 0, i, j;
    for (i = 0; i < n && cnt < 512; i++) ids[cnt++] = k->byte_id[s[i]];
    for (;;) {
        int best = k->nm, at = -1;
        for (i = 0; i + 1 < cnt; i++)
            for (j = 0; j < best; j++)
                if (k->merge[j].a == ids[i] && k->merge[j].b == ids[i + 1]) { best = j; at = i; break; }
        if (at < 0) break;
        ids[at] = k->merge[best].out;
        for (i = at + 1; i + 1 < cnt; i++) ids[i] = ids[i + 1];
        cnt--;
    }
    memcpy(out, ids, cnt * sizeof(int));
    return cnt;
}

/* GPT-2 style pre-tokenization (non-ASCII bytes count as letters), then BPE. */
static int tok_encode(Tok *k, const char *text, int *out) {
    const unsigned char *s = (const unsigned char *)text;
    int n = (int)strlen(text), i = 0, cnt = 0;
    while (i < n) {
        int st = i;
        if (s[i] == '\'' && i + 1 < n && strchr("stmd", s[i + 1]) && !(i + 2 < n && is_letter(s[i + 2]))) i += 2;
        else if (s[i] == '\'' && i + 2 < n && ((s[i+1] == 'r' && s[i+2] == 'e') || (s[i+1] == 'v' && s[i+2] == 'e') || (s[i+1] == 'l' && s[i+2] == 'l'))) i += 3;
        else {
            int j = i + (s[i] == ' ' && i + 1 < n && !is_space(s[i + 1]) ? 1 : 0);
            if (is_letter(s[j])) { i = j; while (i < n && is_letter(s[i])) i++; }
            else if (is_digit(s[j])) { i = j; while (i < n && is_digit(s[i])) i++; }
            else if (!is_space(s[j])) { i = j; while (i < n && !is_space(s[i]) && !is_letter(s[i]) && !is_digit(s[i])) i++; }
            else {  /* whitespace run: leave the last space for the next word */
                while (i < n && is_space(s[i])) i++;
                if (i < n && i - st > 1 && s[i - 1] == ' ') i--;
            }
        }
        cnt += bpe_chunk(k, s + st, i - st, out + cnt);
    }
    return cnt;
}

/* ---------- Mac Roman <-> UTF-8 ---------- */
static const unsigned short macroman[128] = {
    0xC4,0xC5,0xC7,0xC9,0xD1,0xD6,0xDC,0xE1,0xE0,0xE2,0xE4,0xE3,0xE5,0xE7,0xE9,0xE8,
    0xEA,0xEB,0xED,0xEC,0xEE,0xEF,0xF1,0xF3,0xF2,0xF4,0xF6,0xF5,0xFA,0xF9,0xFB,0xFC,
    0x2020,0xB0,0xA2,0xA3,0xA7,0x2022,0xB6,0xDF,0xAE,0xA9,0x2122,0xB4,0xA8,0x2260,0xC6,0xD8,
    0x221E,0xB1,0x2264,0x2265,0xA5,0xB5,0x2202,0x2211,0x220F,0x3C0,0x222B,0xAA,0xBA,0x3A9,0xE6,0xF8,
    0xBF,0xA1,0xAC,0x221A,0x192,0x2248,0x2206,0xAB,0xBB,0x2026,0xA0,0xC0,0xC3,0xD5,0x152,0x153,
    0x2013,0x2014,0x201C,0x201D,0x2018,0x2019,0xF7,0x25CA,0xFF,0x178,0x2044,0x20AC,0x2039,0x203A,0xFB01,0xFB02,
    0x2021,0xB7,0x201A,0x201E,0x2030,0xC2,0xCA,0xC1,0xCB,0xC8,0xCD,0xCE,0xCF,0xCC,0xD3,0xD4,
    0xF8FF,0xD2,0xDA,0xDB,0xD9,0x131,0x2C6,0x2DC,0xAF,0x2D8,0x2D9,0x2DA,0xB8,0x2DD,0x2DB,0x2C7 };

static void mac_to_utf8(const char *in, char *out) {
    const unsigned char *s = (const unsigned char *)in;
    for (; *s; s++) {
        unsigned int u = *s < 0x80 ? *s : macroman[*s - 0x80];
        if (u < 0x80) *out++ = (char)u;
        else if (u < 0x800) { *out++ = (char)(0xC0 | (u >> 6)); *out++ = (char)(0x80 | (u & 0x3F)); }
        else { *out++ = (char)(0xE0 | (u >> 12)); *out++ = (char)(0x80 | ((u >> 6) & 0x3F)); *out++ = (char)(0x80 | (u & 0x3F)); }
    }
    *out = 0;
}

static unsigned char unicode_to_mac(unsigned int u) {
    int i;
    if (u < 0x80) return (unsigned char)u;
    if (u == 0x2019 || u == 0x2018) return '\'';
    for (i = 0; i < 128; i++) if (macroman[i] == u) return (unsigned char)(0x80 + i);
    return '?';
}

/* Prints UTF-8 bytes as Mac Roman (or raw UTF-8 on a native build), buffering split sequences. */
static unsigned char u8buf[4]; static int u8len, u8need;
static void emit_bytes(const unsigned char *b, int n) {
    int i;
    for (i = 0; i < n; i++) {
#ifdef MAC_TARGET
        unsigned char c = b[i];
        if (u8need == 0) {
            if (c < 0x80) { putchar(c); continue; }
            u8need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2; u8len = 0;
        }
        u8buf[u8len++] = c;
        if (u8len == u8need) {
            unsigned int u = u8need == 2 ? ((u8buf[0] & 0x1F) << 6) | (u8buf[1] & 0x3F)
                           : u8need == 3 ? ((u8buf[0] & 0x0F) << 12) | ((u8buf[1] & 0x3F) << 6) | (u8buf[2] & 0x3F)
                           : 0xFFFD;
            putchar(unicode_to_mac(u)); u8need = 0;
        }
#else
        putchar(b[i]);
#endif
    }
    fflush(stdout);
}

/* ---------- sampling ---------- */
static unsigned long rng_state = 88172645UL;
static float frand(void) {
    rng_state ^= rng_state << 13; rng_state &= 0xFFFFFFFFUL;
    rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; rng_state &= 0xFFFFFFFFUL;
    return (rng_state >> 8) / 16777216.0f;
}
typedef struct { float p; int i; } PI;
static int cmp_pi(const void *a, const void *b) { float d = ((const PI *)b)->p - ((const PI *)a)->p; return d > 0 ? 1 : d < 0 ? -1 : 0; }
static PI *pi_buf;

static int sample(float *logits, int V, const int *recent, int nrecent, float temp, float topp, float rep) {
    int i, n = 0; float mx = -1e30f, sum = 0, cum = 0, r;
    for (i = 0; i < nrecent; i++) { float *z = &logits[recent[i]]; if (*z > 1e29f) continue; *z = *z > 0 ? *z / rep : *z * rep; }
    for (i = 0; i < V; i++) { logits[i] /= temp; if (logits[i] > mx) mx = logits[i]; }
    for (i = 0; i < V; i++) { float e = (float)exp(logits[i] - mx); sum += e; pi_buf[i].p = e; pi_buf[i].i = i; }
    /* keep only candidates that can matter for top-p before sorting */
    for (i = 0; i < V; i++) if (pi_buf[i].p / sum > (1.0f - topp) / (V - 1)) pi_buf[n++] = pi_buf[i];
    qsort(pi_buf, n, sizeof(PI), cmp_pi);
    for (i = 0; i < n; i++) { cum += pi_buf[i].p / sum; if (cum >= topp) { n = i + 1; break; } }
    r = frand() * (cum > 0 ? cum : 1);
    for (i = 0, cum = 0; i < n; i++) { cum += pi_buf[i].p / sum; if (r < cum) return pi_buf[i].i; }
    return pi_buf[n - 1].i;
}

static double now_seconds(void) {
#ifdef MAC_TARGET
    return TickCount() / 60.0;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* ---------- chat ---------- */
int main(int argc, char **argv) {
    static Model m; static Tok k;
    static int ids[1024], gen[512];
    static char line[512], utf[1600], prompt[1800];
    int pos = 0, n, i, ng, tok;
    const char *one_shot = argc > 1 ? argv[1] : NULL;   /* native testing: one message, then exit */

    printf("Nicu per Mac OS 9 - caricamento...\n"); fflush(stdout);
    if (!model_load(&m, "nicu.dat")) { printf("Non trovo nicu.dat (o memoria insufficiente).\n"); return 1; }
    if (!tok_load(&k, "nicutok.dat")) { printf("Non trovo nicutok.dat.\n"); return 1; }
    pi_buf = malloc(m.V * sizeof(PI));
    rng_state ^= (unsigned long)(now_seconds() * 60.0) + (unsigned long)time(NULL);
    printf("Nicu e' pronto: %d parametri circa %ldK. Scrivi e premi Invio.\n\n",
           0, (long)m.V * m.C / 1000 * 0 + 5300);

    for (;;) {
        double t0;
        if (one_shot) { strncpy(line, one_shot, sizeof line - 1); printf("Tu: %s\n", line); }
        else {
            printf("Tu: "); fflush(stdout);
            if (!fgets(line, sizeof line, stdin)) break;
        }
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
#ifdef MAC_TARGET
        mac_to_utf8(line, utf);
#else
        strcpy(utf, line);
#endif
        sprintf(prompt, "%sUtente: %s\nNicu:", pos ? "\n" : "", utf);
        n = tok_encode(&k, prompt, ids);
        if (pos + n + 120 >= m.T) {               /* context full: start a fresh conversation */
            pos = 0;
            sprintf(prompt, "Utente: %s\nNicu:", utf);
            n = tok_encode(&k, prompt, ids);
        }
        printf("Nicu:"); fflush(stdout);
        for (i = 0; i < n; i++) forward(&m, ids[i], pos++);
        t0 = now_seconds(); ng = 0;
        for (;;) {
            tok = sample(m.logits, m.V, gen, ng, 1.0f, 0.92f, 1.15f);
            if (tok == 0 || memchr(k.tok[tok], '\n', k.len[tok])) break;   /* end of turn */
            gen[ng++] = tok;
            emit_bytes(k.tok[tok], k.len[tok]);
            if (pos >= m.T - 1 || ng >= 500) break;
            forward(&m, tok, pos++);
        }
        if (tok == 0) pos = 0;                      /* dialogue ended: next turn starts fresh */
        {
            double dt = now_seconds() - t0;
            printf("\n   [%d token in %.1f s, %.1f token/s]\n\n", ng, dt, dt > 0 ? ng / dt : 0.0);
        }
        if (one_shot) break;
    }
    return 0;
}
