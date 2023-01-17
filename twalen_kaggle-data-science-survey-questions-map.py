import re
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
sns.set(style="white")
schema = pd.read_csv("../input/SurveySchema.csv", low_memory=False)
resp = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False)
resp_h, resp = resp.iloc[0], resp.iloc[1:]
resp_freeform = pd.read_csv("../input/freeFormResponses.csv", low_memory=False)
schema.shape, resp.shape, resp_freeform.shape
resp.head(3)
def clean_title(t):
    t = re.sub(r'- .*$', '', t)
    t = re.sub(r'\s+\(Select all that apply\)\s*$', '', t)
    return t

def detect_question_category(t):
    if re.match(r'.*(gender|your age|country).*', t):
        return 'personal'
    elif re.match(r'.*(Your views|Do you consider|How do you perceive|Which better).*', t):
        return 'opinions'
    elif re.match(r'.*(education|undergraduate|data science courses|learn first).*', t):
        return 'education'
    elif re.match(r'.*(current role|current employer|yearly compensation).*', t):
        return 'career'
    elif re.match(r'.*(have you used|programming language|primary tool|tools and methods).*$', t):
        return 'tools'
    return 'other'

def detect_question_type(columns):
    if any(re.match('.*_Part_.*', c) for c in columns):
        c = columns[0]
        values = resp[c].dropna().unique()
        if len(values) > 1:
            return "mcq/text"
        else:
            return "mcq/simple"
    else:
        assert len(columns) == 1
        c = columns[0]
        values = resp[c].dropna().unique()
        assert len(values) > 0
        if isinstance(values[0], str):
            if len(values) < 20:
                return 'categorical/str'
            else:
                return 'str'
        return 'other'
res = []
for num in range(1, 51):
    sel = resp_h.filter(regex=f'Q{num}(_.*)?$')
    sel_other = sel.filter(regex='_OTHER(_TEXT)?$')
    sel_main = sel[~sel.index.isin(sel_other.index)]
    title = clean_title(sel.iloc[0])
    has_other = len(sel_other) > 0
    t = detect_question_type(sel_main.index.values)
    cat = detect_question_category(title)
    
    if re.match('^mcq', t):
        options = [re.sub(r'^.*\s+-\s+(.*)', r'\1', c) for c in sel_main.values]
    else:
        options = []
    
    res.append(OrderedDict([
        ('num', num),
        ('category', cat),
        ('type', t),
        ('columns_count', len(sel_main)),
        ('has_other', has_other),
        ('title', title),
        ('options', options)
    ]))
question_map = pd.DataFrame(res)
question_map
def excel_set_col_width(writer, sheet_name, width, first_column, last_column=None):
    wb = writer.book
    ws = writer.sheets[sheet_name]
    ws.set_column(first_column, last_column if last_column is not None else first_column, width)

fn = 'kaggle-survey-map.xlsx'
writer = pd.ExcelWriter(fn, datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd')
question_map.to_excel(writer, sheet_name="questions", index=False)
for col_num, col_w in [
    (2, 15),
    (3, 12),
    (5, 30),
    (6, 30),
]:
    excel_set_col_width(writer, "questions", col_w, col_num)
writer.close()
fig, ax = plt.subplots(figsize=(15, 0.3))
categories = question_map['category'].unique()
p = sns.color_palette('Set1', len(categories))
d = dict([(y,p[i]) for i, y in enumerate(categories)])
c = question_map['category'].map(d)
ax.bar(x=question_map['num'], height=1, color=c, width=0.9)
ax.grid(False)
ax.set_xticks([1] + list(range(5, len(question_map) + 1, 5)))
ax.yaxis.set_visible(False)
#ax.set_axis_off()
legend_elements = []
for c in question_map['category'].factorize()[1].tolist():
    color = d[c]
    legend_elements.append(
        Patch(facecolor=color, edgecolor=color, label=c)
    )
ax.legend(handles=legend_elements, ncol=len(legend_elements), frameon=False, 
          loc='upper center', bbox_to_anchor=(0.5, -2))
sns.despine(ax=ax, left=True, bottom=True)
plt.show()
