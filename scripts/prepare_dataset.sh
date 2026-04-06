if [ ! -f hedc4-phonemes_v1.txt ]; then
    wget -nc https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/hedc4-phonemes_v1.txt.7z
    7z x hedc4-phonemes_v1.txt.7z
fi

uv run python -c "
import re, unicodedata, sys
with open('hedc4-phonemes.txt', encoding='utf-8') as fin, \
     open('hedc4-hebrew.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        hebrew = line.split('\t')[0]
        hebrew = unicodedata.normalize('NFD', hebrew)
        hebrew = re.sub(r'[\u0590-\u05cf|]', '', hebrew)
        if hebrew:
            fout.write(hebrew + '\n')
"