!pip3 install sacrebleu
import sacrebleu
dev_en_csv = '/kaggle/input/shopee-product-title-translation-open/dev_en.csv'
dev_tcn_csv = '/kaggle/input/shopee-product-title-translation-open/dev_tcn.csv'
df = pd.concat([pd.read_csv(dev_tcn_csv), pd.read_csv(dev_en_csv)], axis=1).drop('split', axis=1)
df.head()
refs = [df['text'], df['translation_output']]
sys = df['translation_output']
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Case Phone Soft Rabbit Silicone Case']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Phone Case Soft Rabbit Silicone Case AddWord']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Phone Phone Case Soft Rabbit Silicone Case AddWord']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 phone Case Soft Rabbit Silicone Case']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['Oppo A75 A75S A73 Case Soft Rabbit Silicone Case']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
refs = [[df.iloc[0]['text']], [df.iloc[0]['translation_output']]]
sys = ['A75 A75S A73 Phone Case Soft Rabbit Silicone']
print(f'refs: {refs}')
print(f'sys: {sys}')
bleu = sacrebleu.corpus_bleu(sys, refs, lowercase=True)
print(bleu.score)
