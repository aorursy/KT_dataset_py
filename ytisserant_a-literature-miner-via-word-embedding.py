!pip install scispacy  --quiet

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz  --quiet

!pip install wmd  --quiet
!pip install scireader==0.0.4 --quiet 
import spacy

import scispacy

import glob

from scireader import *



root_path = '/kaggle/input/CORD-19-research-challenge/'

jsonfiles = glob.glob(f'{root_path}/**/pdf_json/*.json', recursive=True)

print('Load '+str(len(jsonfiles))+" papers for this study.")



nlp = en_core_sci_lg.load()

abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)

nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)
bank=PaperBank(nlp)

bank.read(jsonfiles)

bank.parse('abstract')

print('Done building the PaperBank object')
bank.query('spike protein',similarity=0.8,verbose=True)
sentence1='underlying disease may be a risk factor for the ICU patients'

sentence2='hypertension strongly predictive severe disease admission'

sentence3='these epitopes may potentially offer protection against this novel virus'



similarity12=sentSimilarity(bank,sentence1,sentence2)

similarity13=sentSimilarity(bank,sentence1,sentence3)



print('The similarity score between the sentence 1 and sentence 2 is: '+str(similarity12))

print('The similarity score between the sentence 1 and sentence 3 is: '+str(similarity13))
kw_covid19=['severe acute respiratory syndrome coronavirus.2','cov.19','covid 19','2019 corona virus','corona virus 2019','sars cov 2','2019 cov','cov.2','coronavirus.2','wuhan cov','wuhan corona virus','pandemic corona virus','corona virus pandemic']



kw_therapy=['naproxen','arbidol hydrochloride','oseltamivir','angiotensin converting inhibitor','ace2 inhibitor','ace.2 inhibitor','arbidol','asc09','ritonavir','atazanavir','aviptadil','oseltamivir','azithromycin','baricitinib','bevacizumab','bromhexine hydrochloride','camostat mesilate','carrimycin','cd24fc','chloroquine diphosphate','chloroquine','hydroxychloroquine','chloroquine phosphate','colchicine','alfa.*interferon','interferon alfa','lopinavir','ritonavir','darunavir','cobicistat','das181','dexamethasone','eculizumab','escin','favipiravir','tocilizumab','fingolimod','hydrocortisone','ceftriaxone','moxifloxacin','levofloxacin','piperacillin tazobactam','piperacillin','tazobactam','ceftaroline','amoxicillin','clavulanate', 'amoxicillin clavulanate','macrolide','oseltamivir','interferon β1a','interferon beta','anakinra','ganovo','danoprevir','huaier granule','hyperbaric oxygen','losartan','meplazumab','methylprednisolone','acetylcysteine','nitric oxide','pd.1 antibody','thymosin','thalidomide','plaquenil','pul.042','pul042', 'rhACE2','recombinant angiotensin converting enzyme','recombinant human interferon alpha','thymosin alpha','recombinant human interferon','remdesivir','roactemra','kevzara','sargramostim','sarilumab','sildenafil citrate','tetrandrine','tocilizumab','clarithromycin','minocyclin','randomize control trial', 'odds ratio', 'observation case series', 'randomized trial', 'case.control', 'interrupt time.series', 'hazard ratio', 'odds ratio', 'treatment effect','rate adverse event', "reduction disease symptom","reduction symptom"]



hits_covid19=scanPapersByKW(bank,kw_covid19,similarity_outcome=0.9)



hits_therapy=scanPapersByKW(bank,kw_therapy,similarity_outcome=0.99)



candidates_therapy=list(set(hits_covid19).intersection(set(hits_therapy)))



print('Found '+str(len(hits_covid19))+' papers by semantic search of keywords related to COVID-19')

print('Found '+str(len(kw_therapy))+' papers by semantic search of keywords related to candidate therapeutics')

print('Jointly we found '+str(len(candidates_therapy))+' papers that are related to both COVID-19 and therapeutics')

inquiry='evaluation of covid-19 treatment'

answer=scanPapersBySent(bank,candidates_therapy,[inquiry],distance=3)



print('Top 20 papers (id) and their WMD scores for the inquiry (\''+inquiry+'\'):')

answer[inquiry][0:20]
from scireader.utils import cleanText

for hit in answer[inquiry][0:20]:

    print('paper id='+hit[0],'score='+str(hit[1]))

    print(cleanText(bank.text.loc[hit[0],'abstract']))

    print('\n')