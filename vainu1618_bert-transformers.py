!pip install pytorch_transformers
!pip install pymorphy2
!pip install autocorrect
import os
import re
import time
import random
import datetime

import nltk
import torch
import pymorphy2
import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW
from autocorrect import Speller
class DataPreparation():
    
    def __init__(self):
        
        #
        self.stop_words = None
        self.stop_words_excep = ['интернет', 'оператор', 'тариф', 'интернета',
                                 'работает', 'нет']
        
        self.stop_words_extra = ['добрый', 'доброго', 'добры', 'доброй', 'доброе', 'добрые',
                                 'здравствуйте', 'здрасте', 'здравчтвуйтет', 'здрастивити',
                                 'здравствуй', 'здрастувуйте', 'здраствутйте', 'здрсти', 
                                 'здрьте', 'вопросздравствуйте', 'введитездравствуйте', 
                                 'здрасьте', 'здрастуйте', 'здраастауйте', 'введздравствуйте',
                                 'здравсте', 'привет', 'приветствую', 'или', 'который',
                                 'очень', 'еще', 'это']
        
        # Словарь английских слов
        nltk.download('words')
        self.vocab_en = set(w.lower() for w in words.words())
        
        
    def load_csv(self, path_in, sep=',', encoding='utf-8', engine='python'):
            """ Загружаем CSV-файл
            """
            if not engine == 'python':
                engine = None 
            return  pd.read_csv(path_in, sep=sep, encoding=encoding, engine=engine)
    
    
    def save_csv(self, df, path_out, sep=',', encoding='utf-8', index=False, header=True):
            """ Сохраняем CSV-файл
            """
            path_out = path_out.replace('\\','/')
            if not os.path.exists('/'.join(path_out.split('/')[0:-1])):
                os.makedirs('/'.join(path_out.split('/')[0:-1]))
            df.to_csv(path_out, index=index, header=header, sep=sep, encoding=encoding) 
    
    
    def fill_na(self, df, column_str, replace_to=''):
            """ Заполнение пропусков
            """
            df[column_str] = df[column_str].fillna(replace_to)
            
    def clear_chars(self, text):
            """ Удаляем лишние символы, включая лишние пробелы и переносы строк
            """
            text = re.sub('[^a-zA-Zа-яА-Я 0-9]+', ' ', text) 
            return ' '.join(text.split())
    
    def do_lower(self, text):
            """ Приводим текст к нижнему регистру
            """
            return text.lower()
    
    
    def replace_empty_to_max_freq_label(self, df, text_str, label_str, empty_str=''):
            """ Определяем наиболее частую метку для пропущенных данных
                и присваиваем её для всех пропусков.
            """
            df_empty = df[df[text_str] == empty_str]
            df_empty_uniq = pd.value_counts(df_empty.values.ravel())[1:]
            try:
                max_freq_label = df_empty_uniq.index[0]
                df_empty_indx = df_empty.index.to_list()
                for empty_indx in df_empty_indx:
                    df.loc[empty_indx,label_str] = max_freq_label
            except:
                pass
    
    
    def tokenize_by_rules(self, string):
            """ Токенизируем строку по заданной группе(правилам разделения)
            """
            token = ''
            tokens = []
            category = None
            categories = ['0123456789',
                          'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                          'abcdefghijklmnopqrstuvwxyz']
            for char in string:
                if token:
                    if category and char.lower() in category:
                         token += char
                    else:
                         if not token == ' ':
                             tokens.append(token)
                         token = char
                         category = None
                         for cat in categories:
                             if char.lower() in cat:
                                 category = cat
                                 break
                else:
                     category = None
                     if not category:
                         for cat in categories:
                             if char.lower() in cat:
                                 category = cat
                                 break
                     token += char
            if token:
                 if not token == ' ':
                     tokens.append(token)
            return ' '.join(tokens)
    
        
    def fix_text_case(self, text_etalon, text_current, fix_aggressive=False):
            """ Сравниваем два текста и пытаемся восстановить регистр ()
                • aggressive - если False, то меняем только первый символ; True - все символы
            """
            
            if not fix_aggressive:
                
                text_etalon = text_etalon.split()
                text_current = text_current.split()
                if not len(text_etalon) == len(text_current):
                    return ' '.join(text_current)
    
                i = 0
                text_new = text_current
                for t_bef,t_aft in zip(text_etalon,text_current):
                    case_etalon = 'Upper' if t_bef[0].istitle() else 'Lower'
                    case_current = 'Upper' if t_aft[0].istitle() else 'Lower'
            
                    if not case_etalon == case_current: 
                        if case_etalon == 'Upper' and case_current == 'Lower':
                            text_new[i] = t_aft[0].upper() + t_aft[1:]
                        elif case_etalon == 'Lower' and case_current == 'Upper':
                            text_new[i] = t_aft[0].lower() + t_aft[1:]
                    i += 1
                return ' '.join(text_new)
    
            else:
                
                text_new = ''
                if not len(text_etalon) == len(text_current):
                    return text_current
                
                for t_bef,t_aft in zip(text_etalon,text_current):
                    case_etalon = 'Upper' if t_bef.istitle() else 'Lower'
                    case_current = 'Upper' if t_aft.istitle() else 'Lower'
            
                    if not case_etalon == case_current: 
                        if case_etalon == 'Upper' and case_current == 'Lower':
                            text_new += t_aft.upper()
                        elif case_etalon == 'Lower' and case_current == 'Upper':
                            text_new += t_aft.lower()
                    else:
                        text_new += t_aft
                return text_new
            
    def do_lemmatization(self, text, fix_case=True, fix_aggressive=False):
            """ Лемматизатор для русских текстов (pymorphy)
            """
            new_sentence = [self.morph_ru.parse(word)[0].normal_form for word in text.split()]
            if fix_case:
                fix_sentence = []
                old_sentence = text.split()
                for old,new in zip(old_sentence,new_sentence):
                    fix_sentence.append(self.fix_text_case(old, new, fix_aggressive))
                return ' '.join(fix_sentence) 
            return ' '.join(new_sentence)
    
    
    def fix_spell(self,text):
        return spell(text.lower())
        
    
    
    def creat_stop_words(self, df, text_str, label_str, 
                         stop_words_excep, stop_words_extra, min_labels_count=None):
            """ Формируем список стоп-слов (слова которые встречаются в заданном числе меток)
                • min_labels_count - минимальное кол-во категорий,
                                     в которых должно содержаться стоп-слово
            """
    
            # Список меток
            labels = list(set(df[label_str].to_list()))
            
            # Кол-во категорий в которых должно содержаться стоп-слово
            if min_labels_count is None:
                min_labels_count = len(labels)
    
            #Список слов для каждой метки
            labels_word = {}
            for label in labels:
                labels_word[label]= self.get_dict(df[df[label_str] == label], text_str)
              
            # Формируем список стоп-слов  
            stop_words = []
            uniq_words = self.get_dict(df, text_str)
              
            for word in uniq_words:
                count = 0
                for label in labels:
                    if word in labels_word[label]:
                        count += 1
                if count >= min_labels_count and word not in stop_words_excep:
                    stop_words.append(word)
            
            stop_words.extend(stop_words_extra)
            return stop_words 
            
    def delete_stopwords(self, text):
            """ Очистка от стоп-слов
                • stops - список стоп слов. Если пустой то используем стандартные
            """
            if self.stop_words is None:
                self.stop_words = set(stopwords.words("english")) | set(stopwords.words("russian"))
            new_sentence = [word for word in text.split() if not word.lower() in self.stop_words]
            return ' '.join(new_sentence)    
        
    def word_is_en(self, word):
            return word.lower() in self.vocab_en
    
    def word_is_ru(self, word):
            return self.morph_ru.word_is_known(word.lower(), strict_ee=False)
    
    def get_dict(self, df, column_str='', uniq=True):
            """ Список уникальных слов в конкретном столбце
            """
            if not isinstance(df,list):
                rows = df[column_str].to_list()
            else:
                rows = df
            dictionary = []
            for row in rows:
                dictionary.extend(row.split())
            return list(set(dictionary)) if uniq else dictionary        



    def preprocessing(self, do_clen, do_lower_case, replace_empty_label,
                      do_tokenize, drop_dupl, do_lemm, del_stops,fix_spelling):
    
        # Заменяем пропуски
        self.fill_na(csv_train, 'text', replace_to='')
        self.fill_na(csv_test, 'text', replace_to='')
        
        # Удаляем лишние символы
        if do_clen:
            print('Удаляем лишние символы..')
            csv_train['text'] = csv_train['text'].apply(self.clear_chars)
            csv_test['text'] = csv_test['text'].apply(self.clear_chars)
        
        # Переводим в нижний регистр
        if do_lower_case:
            print('Переводим в нижний регист..')
            csv_train['text'] = csv_train['text'].apply(self.do_lower)
            csv_test['text'] = csv_test['text'].apply(self.do_lower)
        
        # Определяем наиболее частую метку для пропусков 
        # и присваиваем её для всех пропусков в тренировочных данных
        if replace_empty_label:
            print('Заменяем пустые метки..')
            self.replace_empty_to_max_freq_label(csv_train, text_str='text', 
                                                 label_str='label', empty_str='')

        # Токенизируем текст (отделяем числа от текста и английский текст от русского)
        if do_tokenize:
            print('Токенизация..')
            csv_train['text'] = csv_train['text'].apply(self.tokenize_by_rules)
            csv_test['text'] = csv_test['text'].apply(self.tokenize_by_rules)
            
        # Исправление опечаток
        if fix_spelling:
            csv_train['text'] = csv_train['text'].apply(self.fix_spell)
            csv_test['text'] = csv_test['text'].apply(self.fix_spell)            
        
        # Лемматизация
        if do_lemm:
            print('Лемматизация..')
            self.morph_ru = pymorphy2.MorphAnalyzer() # Экземпляр класса pymorphy2
            csv_train['text'] = csv_train['text'].apply(self.do_lemmatization)
            csv_test['text'] = csv_test['text'].apply(self.do_lemmatization) 
            
        # Формируем список стоп-слов (слова которые встречаются в заданном числе меток)
        if del_stops:
            print('Формируем список стоп-слов..')
            self.stop_words = self.creat_stop_words(csv_train, 'text', 'label',
                                                    self.stop_words_excep,
                                                    self.stop_words_extra,
                                                    min_labels_count=None)
        
            # Удаляем стоп-слова
            print('Удаляем стоп-слова..')
            csv_train['text'] = csv_train['text'].apply(self.delete_stopwords)
            csv_test['text'] = csv_test['text'].apply(self.delete_stopwords)    
            
            
        # Удаляем дубликаты  
        if drop_dupl:
            print('Удаляем дубликаты..')
            csv_train.drop_duplicates(inplace=True)
        
        # Заменяем пустые строки на пробел (фикс ошибки при обучении BERT)
        print('Заменяем пустые строки на пробел..')
        csv_train['text'] = csv_train['text'].replace('', ' ')
        csv_test['text'] = csv_test['text'].replace('', ' ')   
        
        variant = 'clen'+str(do_clen)[0]+\
                  '_replaceempty'+str(replace_empty_label)[0]+\
                  '_tokenized'+str(do_tokenize)[0]+\
                  '_lower'+str(do_lower_case)[0]+\
                  '_stops'+str(del_stops)[0]+\
                  '_dropdupl'+str(drop_dupl)[0]+\
                  '_lemm'+str(do_lemm)[0]+\
                  '_fix_spelling'+str(fix_spelling)[0]
            
        return variant
class EarlyStopping:

    def __init__(self, path, patience=1, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        save_pretrained_model(model,self.path)         
        self.val_loss_min = val_loss
def check_device():

    # If there's a GPU available...
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        device_type = 'gpu'
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu") 
        device_type = 'cpu'
        
    return device,device_type
    
    
def set_seed(seed_value=42):
    """ Задаем seed для воспроизведения результаов
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    


def check_BPE_maxlen(sentences,print_str=''):
    """ Находим максимальное количество токенов в списке из предложений
    """
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    print('Max '+print_str+' length: ', max_len)
    return max_len


def get_tokenized_tensors(sentences, labels=None, max_length=512, show_example=False):
    """ Токенезируем все предложения и сопоставляем токены с их id в словаре.
    """
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = max_length,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
        )
        
        # Создаем список из последовательностей чисел, 
        # отождествляющих каждый токен с его номером в словаре.    
        input_ids.append(encoded_dict['input_ids'])
        
        # Список из последовательностей нулей и единиц, 
        # где единицы обозначают токены предложения, нули - паддинг.
        # Паддинг нужен для того, чтобы BERT мог работать с предложениями разной длины.
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Конвертируем списки в pytorch тензоры
    tensor_input_ids = torch.cat(input_ids, dim=0)
    tensor_attention_masks = torch.cat(attention_masks, dim=0)
    tensor_labels = torch.tensor(labels) if not labels is None else None
        
    
    # Выводим пример (по требованию)
    if show_example:
        print('Original: ', sentences[0])
        print('Token IDs:', input_ids[0])
    
    return tensor_input_ids,tensor_attention_masks,tensor_labels



def creat_dataloader(train_dataset, valid_dataset, batch_size=32):
    """  Создаем генератор данных
    """
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), # SequentialSampler
                batch_size = batch_size 
    )
    
    valid_dataloader = DataLoader(
                valid_dataset, 
                sampler = SequentialSampler(valid_dataset), 
                batch_size = batch_size 
    )
    
    return train_dataloader, valid_dataloader




def get_skf_dataloader(sentences, labels, batch_size=32):
    """ Производим разбиение данных (StratifiedKFold) на гурппы
        и возвращаем генераторы данных для каждой из созданных групп.
    """
    for train_index, val_index in skf.split(sentences,labels):
        X_train = [sentences[i] for i in train_index]
        Y_train = [labels[i] for i in train_index]
        X_valid = [sentences[i] for i in val_index]
        Y_valid = [labels[i] for i in val_index]
        

        # Токенизируем и конвертируем в pytorch тенсоры: Тренировочные данные
        train_input_ids,train_attention_masks,train_labels = get_tokenized_tensors(
                                                         sentences = X_train,
                                                         labels = Y_train,
                                                         max_length = MAX_LEN,
                                                         show_example = False) 
        
        # Токенизируем и конвертируем в pytorch тенсоры: Валидационные данные
        valid_input_ids,valid_attention_masks,valid_labels = get_tokenized_tensors(
                                                         sentences = X_valid,
                                                         labels = Y_valid,
                                                         max_length = MAX_LEN,
                                                         show_example = False)
        # Создаем генераторы данных
        train_dataset = TensorDataset(train_input_ids, 
                                      train_attention_masks,
                                      train_labels)
        
        valid_dataset = TensorDataset(valid_input_ids,
                                      valid_attention_masks, 
                                      valid_labels)
        
        train_dataloader, valid_dataloader = creat_dataloader(train_dataset,
                                                              valid_dataset,
                                                              batch_size = batch_size)
        
        yield train_dataloader, valid_dataloader
        



def load_pretrained_model(model_name):
    """ Загружаем предобученную BERT модель
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels = NUM_LABELS)
    
    if device_type == 'gpu':
        model.cuda()
    return model



def prepare_optimizer(lr=2e-5, eps=1e-8, weight_decay=0.01):
    """ Настраиваем оптимизатор
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    
    return optimizer



def prep_scheduler(num_warmup_steps=0):
    """ Создаем и настраиваем планировщик скорости обучения
    """
    # Общее количество шагов обучения
    total_steps = len(train_dataloader) * EPOCHS
    
    # Создаем планировщик скорости обучения
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = num_warmup_steps,
                                                num_training_steps = total_steps)
    return scheduler


def get_f1_score(preds, labels, average='micro'):
    """ Рассчитываем метрику F1-score
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(pred_flat, labels_flat, average=average)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))



def train():
    # На основе:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py
    
    # Список для хранения статистики обучения
    training_stats = []
    
    # Засекаем время выполнения всех эпох
    total_t0 = time.time()
    
    for epoch_i in range(0, EPOCHS):
        
        # ========================================
        #               Training
        # ========================================
        print('\n======= KFold {:} / {:} ======='.format(current_kfold, CROSS_VALID_FOLDS))
        print('======= Epoch {:} / {:} ======='.format(epoch_i + 1, EPOCHS))
        print('Training...')
    
        # Засекаем время выполнения одной эпохи
        t0 = time.time()
    
        # Обнуляем ошибку на каждой эпохе
        total_train_loss = 0
    
        # Переводим модель в режим обучения
        model.train()
    
        for step, batch in enumerate(train_dataloader):
            # Выводим логи каждые 40 батчей
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0) # время выполнения
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
            # Добавляем батч для вычисления на GPU
            batch = tuple(t.to(device) for t in batch)
    
            # Распаковываем данные из dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # Обнуляем ранее расчитанные градиенты
            model.zero_grad()        
    
            # Прямой проход. Расчитываем логиты
            loss, logits = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)
    
            # Суммируем ошибку по всем батчам
            total_train_loss += loss.item()
    
            # Выполняем обратный проход, чтобы вычислить градиенты.
            loss.backward()
    
            # Боремся со "взрывающимися градиентами"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            # Обновляем параметры с учетом рассчитанного градиента
            optimizer.step()
    
            # Обновляем learning rate.
            scheduler.step()
    
        # Седняя ошибка на одной эпохе
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Время обучения одной эпохи
        training_time = format_time(time.time() - t0)
    
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoсh took: {:}".format(training_time))
        
        
        
        # ========================================
        #               Validation
        # ========================================
        print("\nRunning Validation...")
    
        t0 = time.time()
    
        # Выводим модель из режима обучения
        model.eval()
    
        # Отслеживаемые переменные 
        total_eval_f1 = 0
        total_eval_loss = 0
    
        
        for batch in valid_dataloader:

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # Указание модели не вычислять и не сохранять градиенты
            with torch.no_grad():        
    
                # Прямой проход. Расчитываем логиты
                (loss, logits) = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                
            # Суммируем ошибку по всем батчам
            total_eval_loss += loss.item()
    
            # Переводим логиты и метки на CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
    
            # Расчитываем метрику качества и суммируем её по всем батчам
            total_eval_f1 += get_f1_score(logits,label_ids)
            
            
    
        
        # Метрика качества на одной эпохе
        avg_val_f1 = total_eval_f1 / len(valid_dataloader)
        print("  F1-score: {0:.2f}".format(avg_val_f1))
    
        # Ошибка на одной эпохе
        avg_val_loss = total_eval_loss / len(valid_dataloader)
        
        # Время выполнения одной эпохи
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        
        # Сохраняем статистику
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. F1-score.': avg_val_f1,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        
        # Проверка на раннюю остановку. Сохраняем лучшую модель для каждого фолда
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            break
    
    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats



def predict_on_test_set(model,prediction_dataloader):
    """ Делаем предсказания для тестовых данных
    """
    
    # Выводим модель из режима обучения
    model.eval()
    
    # Отслеживаемые переменные  
    predictions,predictions_flat = [],[]
    

    for n,batch in enumerate(prediction_dataloader):
      print('\r[%d]'%(n), end="",flush=True)

      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask = batch
      
      # Указание модели не вычислять и не сохранять градиенты
      with torch.no_grad():
          # Прямой проход. Расчитываем логиты
          outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
    
      logits = outputs[0]
    
      # Переводим логиты на CPU
      logits = logits.detach().cpu().numpy()
    
      # Получаем предсказание метки
      predictions_flat = np.argmax(logits, axis=1).flatten()
    
      # Сохраняем предсказанную метку
      predictions.extend(predictions_flat)
    
    print('    DONE.')
    return predictions


def save_pretrained_model(model,output_dir):
    """ Сохраняем обученную модель
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_result(output_file):
    """ Сохраняем предсказания модели
    """
    def get_test_labels_id(x):
        for index,row in df_labels.iterrows():
            if x == row['label_id']:
                return df_labels.loc[index, 'label']

    # Словарь меток и их id
    df_labels = csv_train[['label', 'label_id']].drop_duplicates()

    # Формируем выходной DataFrame        
    result = pd.DataFrame()   
    result['id'] = csv_test['id']
    result['label'] = predictions
    result['label'] = result['label'].apply(get_test_labels_id)

    # Сохраняем
    DP.save_csv(result, output_file)

    print('Result csv-file was saved to:' + output_file)
# Устройство
device, device_type = check_device()

# Параметры
SEED = 42
BERT_PRETRAINED_MODEL = "DeepPavlov/rubert-base-cased-conversational"
CROSS_VALID_FOLDS = 5
BATCH_SIZE = 32
BATCH_SIZE_TEST = 32  
EPOCHS = 3
LR = 2e-5 
EPS = 1e-8
WEIGHT_DECAY = 0.01
NUM_WARMUP_STEPS = '10%' # n, 'n%' 

# Задаем seed для воспроизводимости результатов
set_seed(SEED)
# Создаем экземпляр класса для предобработки данных
DP = DataPreparation()


spell = Speller(lang='ru')


# Загружаем датасет
csv_train = DP.load_csv("/kaggle/input/ocrv-intent-classification/train.csv")
csv_train = csv_train[['text', 'label']]
csv_test = DP.load_csv("/kaggle/input/ocrv-intent-classification/test.csv")
csv_test = csv_test[['id', 'text']] 


# Предобработка данных
data_prep_settings = DP.preprocessing(
                                    do_clen = False,
                                    do_lower_case = False,
                                    replace_empty_label = True,
                                    do_tokenize = False,
                                    drop_dupl = False,
                                    do_lemm = False,
                                    del_stops = False,
                                    fix_spelling = True
                                    )
# Извлекаем список предложений
sentences = csv_train['text'].values

# Извлекаем список меток (переводим метки в категории)
csv_train['label_id'] = pd.factorize(csv_train['label'])[0]
labels = csv_train['label_id'].values
NUM_LABELS = len(set(labels))
# Загружаем BERT токенайзер
tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL,
                                          do_lower_case = False)

# Определяем максимальное количество токенов в предложении
MAX_LEN_train = check_BPE_maxlen(sentences, print_str='train')
MAX_LEN_test = check_BPE_maxlen(csv_test['text'].values, print_str='test')
MAX_LEN = min(512,max(MAX_LEN_train, MAX_LEN_test))
print('MAX_LEN =', MAX_LEN)
# Кросс-валидация
skf = StratifiedKFold(n_splits=CROSS_VALID_FOLDS,
                      shuffle=True,
                      random_state=SEED)

# Генератор данных с учетом разбивки на группы
skf_dataloader = get_skf_dataloader(sentences,
                                    labels,
                                    batch_size=BATCH_SIZE)
# Формируем строку с вариантом модели
training_settings = 'bs'+str(BATCH_SIZE)+\
                    '_ep'+str(EPOCHS)+\
                    '_lr'+str(LR)+\
                    '_eps'+str(EPS)+\
                    '_wd'+str(WEIGHT_DECAY)+\
                    '_nws'+str(NUM_WARMUP_STEPS)

model_variant = data_prep_settings + '__' + training_settings





current_kfold = 0
training_stats = []
for train_dataloader, valid_dataloader in skf_dataloader:
    current_kfold += 1
    
    # Загружаем предобученную BERT модель
    model = load_pretrained_model(BERT_PRETRAINED_MODEL)
    
    # Настраиваем оптимизатор
    optimizer = prepare_optimizer(lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)

    # Создаем планировщик скорости обучения
    if '%' in str(NUM_WARMUP_STEPS):
        NUM_WARMUP_STEPS = int(EPOCHS * len(train_dataloader) * float(NUM_WARMUP_STEPS.split('%')[0])/100)
    scheduler = prep_scheduler(num_warmup_steps = NUM_WARMUP_STEPS)

    # Класс для ранней остановки обучения
    early_stopping = EarlyStopping(path = '/kaggle/working/models/'+model_variant+'_kfold'+str(current_kfold),
                                   patience = 1, 
                                   delta = 0)
    
    # Обучаем модель
    train_stats = train()
    
    # Собираем статисткиу обучения
    training_stats.append(train_stats)
# DataFrame со статистикой обучения
df_stats_total = []
pd.set_option('precision', 2)
for stats in training_stats:
    df_stats = pd.DataFrame(data=stats)
    df_stats = df_stats.set_index('epoch')
    df_stats_total.append(df_stats)

# Вариант модели
print('Model variant:',model_variant)

# Средняя метркиа на валидационных данных по всем разбиениям датасета
f1_cum = 0
for df_stats in enumerate(df_stats_total):
    f1_cum += float(df_stats[1].iloc[-1,2])
print('Avg. valid. F1-score:',f1_cum/len(df_stats_total))

# Средняя ошибка на валидационных данных по всем разбиениям датасета
loss_cum = 0
for df_stats in enumerate(df_stats_total):
    loss_cum += float(df_stats[1].iloc[-1,1])
print('Avg. valid. loss:',loss_cum/len(df_stats_total))
sentences_test = csv_test['text'].values

# Токенизируем и конвертируем в pytorch тенсоры: Тестовые данные
input_ids_test,attention_masks_test,_ = get_tokenized_tensors(sentences_test,
                                                              max_length=MAX_LEN)

# Создаем загрузчик данных
prediction_data = TensorDataset(input_ids_test, attention_masks_test)
prediction_dataloader = DataLoader(prediction_data, sampler=SequentialSampler(prediction_data), batch_size=BATCH_SIZE_TEST)


for k in range(1,CROSS_VALID_FOLDS+1):
    # Загружаем сохраненную BERT модель (каждый вариант разбивки датасета)
    model_path = model_variant + '_kfold' + str(k)
    model = load_pretrained_model('/kaggle/working/models/'+model_path)

    # Получаем и сохраняем предсказания
    predictions = predict_on_test_set(model,prediction_dataloader)
    save_result('/kaggle/working/result_' + model_path + '.csv')


    
  
# # Находим 2 модели: с наименьшей ошибкой и наибольшей точностью
# best_kfold_loss_index = 0
# best_kfold_score_index  = 0
# loss_min = np.inf
# score_max = 0

# for i,stats in enumerate(df_stats_total):
#     loss = stats.iloc[-1,1]
#     if loss < loss_min:
#         loss_min = loss
#         best_kfold_loss_index = i
        
#     score = stats.iloc[-1,2]
#     if score > score_max:
#         score_max = score
#         best_kfold_score_index = i

# best_model_by_loss = model_variant + '_kfold' + str(best_kfold_loss_index)
# best_model_by_score = model_variant + '_kfold' + str(best_kfold_score_index)
# if best_model_by_loss == best_model_by_score:
#     print('Best model:', best_model_by_loss)
# else:
#     print('Best model by loss:', best_model_by_loss)
#     print('Best model by score:', best_model_by_score)



# if best_model_by_loss == best_model_by_score:
#     # Загружаем лучшию из сохраненных BERT моделей
#     model = load_pretrained_model('/kaggle/working/models/'+best_model_by_loss)

#     # Получаем и сохраняем предсказания
#     predictions = predict_on_test_set(model,prediction_dataloader)
#     save_result('/kaggle/working/result_' + best_model_by_loss + '.csv')
# else:

#     # Загружаем модель с наименьшей ошибкой. Получаем и сохраняем предсказания
#     model = load_pretrained_model('/kaggle/working/models/'+best_model_by_loss)
#     predictions = predict_on_test_set(model,prediction_dataloader)
#     save_result('/kaggle/working/result_bestloss_' + best_model_by_loss + '.csv')
    
#     # Загружаем модель с наибольшей точность. Получаем и сохраняем предсказания
#     model = load_pretrained_model('/kaggle/working/models/'+best_model_by_score)
#     predictions = predict_on_test_set(model,prediction_dataloader)
#     save_result('/kaggle/working/result_bestscore_' + best_model_by_score + '.csv')

