import collections
import sys

PAD = '<pad>'
PAD_ID = 0
UNKNOWN = '<unk>'
UNKNOWN_ID = 1
GO = '<go>'
GO_ID = 2
EOS = '<eos>'
EOS_ID = 3

NEWLINE = '<newline>'
TAB = '<tab>'

def get_char_counts(text_iter):
    chars = collections.Counter()
    for text in text_iter:
        chars.update(ch for ch in text)

    return chars

mapping = {
    '\n': NEWLINE,
    '\t': TAB,
}
def escape_whitespace_char(char):
    return mapping.get(char, char)

inv_mapping = {v: k for k, v in mapping.items()}
def unescape_whitespace_char(char):
    return inv_mapping.get(char, char)

def create_char_vocab(text_iter, outfile):
    chars = get_char_counts(text_iter)
    all_chars = (PAD, UNKNOWN, GO, EOS) + next(zip(*chars.most_common()))
    with open(outfile, 'w', encoding='utf8') as fp:
        for char in all_chars:
            char = escape_whitespace_char(char)
            fp.write(char)
            fp.write('\n')

class Transform(object):
    def __init__(self, vocab_filename):
        self.chars = []
        self.char2id = {}
        with open(vocab_filename, encoding='utf8') as fp:
            for i, line in enumerate(fp):
                line = unescape_whitespace_char(line.rstrip('\n'))
                self.chars.append(line)
                self.char2id[line] = i

    def apply(self, text):
        for ch in text:
            if ch not in self.char2id:
                print("UNKNOWN CHARACTER:", ch)
                print("LEN:", len(ch))
                print("repr:", repr(ch))
                sys.exit(1)

        return [self.char2id.get(ch, self.UNKNOWN_id()) for ch in text]

    def get(self, char):
        return self.char2id.get(char, self.UNKNOWN_id())

    def GO_id(self):
        return self.char2id[GO]

    def EOS_id(self):
        return self.char2id[EOS]

    def PAD_id(self):
        return self.char2id[PAD]

    def UNKNOWN_id(self):
        return self.char2id[UNKNOWN]
