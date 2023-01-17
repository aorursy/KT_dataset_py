import torch





def get_device():

    # Если в системе есть GPU ...

    if torch.cuda.is_available():

        # Тогда говорим PyTorch использовать GPU.

        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # Если нет GPU, то считаем на обычном процессоре ...

    else:

        print('No GPU available, using the CPU instead.')

        device = torch.device("cpu")

    return device





device = get_device()
!pip install wget



def download_dataset():

    import wget

    import os

    import zipfile



    print('Downloading dataset...')

    # URL до zip-файла который содержит датасет.

    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    out_file = './cola_public_1.1.zip'



    # Скачиваем файл (только в случае если не скачали раньше)

    if not os.path.exists(out_file):

        wget.download(url, out_file)

    # Unzip

    if not os.path.exists('./cola_public/'):

        with zipfile.ZipFile(out_file, 'r') as zip_ref:

            zip_ref.extractall(os.path.dirname(out_file))

    print('Complete')





download_dataset()
def get_sentences_and_labels():

    import pandas as pd



    # Загружаем dataset в pandas dataframe.

    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])



    # Выводим число тренировочных предложений.

    print('Number of training sentences: {:,}\n'.format(df.shape[0]))



    # Выводим случайные 10 рядов из таблички.

    print(df.sample(10))



    # Выводим 5 грамматически неверных предложений.

    print(df.loc[df.label == 0].sample(5)[['sentence', 'label']])



    sentences = df['sentence'].values

    labels = df['label'].values



    # Возвращаем все предложения и разметку к ним.

    return sentences, labels





sentences, labels = get_sentences_and_labels()
from transformers import BertTokenizer

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



sentence_number = 0

# Напечатать оригинальное предложение.

print('Original:', sentences[sentence_number])

# Напечатать предложение разбитое на отдельные токены из словаря.

print('Tokenized: ', tokenizer.tokenize(sentences[sentence_number]))

# Напечатать предложение разбитое на номера токенов в словаре.

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[sentence_number])))

max_len = 0

# Считаем какой максимальный размер имеет предложение разбитое на токены и разбавленное спец. токенами.

for sent in sentences:

    # Токенизируем текст и добавляем `[CLS]` и `[SEP]` токены.

    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Обновляем максимум.

    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
input_ids, attention_masks = [], []



# Для всех предложений...

for sent in sentences:

    encoded_dict = tokenizer.encode_plus(

        sent,  # Текст для токенизации.

        add_special_tokens=True,  # Добавляем '[CLS]' и '[SEP]'

        max_length=64,  # Дополняем [PAD] или обрезаем текст до 64 токенов.

        pad_to_max_length=True,

        return_attention_mask=True,  # Возвращаем также attn. masks.

        return_tensors='pt',  # Возвращаем в виде тензоров pytorch.

    )



    # Добавляем токенизированное предложение в список

    input_ids.append(encoded_dict['input_ids'])

    # И добавляем attention mask в список

    attention_masks.append(encoded_dict['attention_mask'])



# Конвертируем списки в полноценные тензоры Pytorch.

input_ids = torch.cat(input_ids, dim=0)

attention_masks = torch.cat(attention_masks, dim=0)

labels = torch.tensor(labels)



# Печатаем предложение с номером 0, его токены (теперь в виде номеров в словаре) и.т.д.

print('Original: ', sentences[0])

print('Token IDs:', input_ids[0])

print('Attention masks:', attention_masks[0])

print('Labels:', labels[0])

from torch.utils.data import TensorDataset, random_split



# Объединяем все тренировочные данные в один TensorDataset.

dataset = TensorDataset(input_ids, attention_masks, labels)



# Делаем разделение случайное разбиение 90% - тренировка 10% - валидация.



# Считаем число данных для тренировки и для валидации.

train_size = int(0.9 * len(dataset))

val_size = len(dataset) - train_size



# Разбиваем датасет с учетом посчитанного количества.

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



print('{:>5,} training samples'.format(train_size))

print('{:>5,} validation samples'.format(val_size))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



# DataLoader должен знать размер батча для тренировки мы задаем его здесь.

# Размер батча – это сколько текстов будет подаваться на сеть для вычисления градиентов

# Авторы BERT предлагают ставить его 16 или 32. 

batch_size = 32



# Создаем отдельные DataLoaders для наших тренировочного и валидационного наборов



# Для тренировки мы берем тексты в случайном порядке.

train_dataloader = DataLoader(

        train_dataset,  # Тренировочный набор данных.

        sampler = RandomSampler(train_dataset), # Выбираем батчи случайно

        batch_size = batch_size # Тренируем с таким размером батча.

)



# Для валидации порядок не важен, поэтому зачитываем их последовательно.

validation_dataloader = DataLoader(

        val_dataset, # Валидационный набор данных.

        sampler = SequentialSampler(val_dataset), # Выбираем батчи последовательно.

        batch_size = batch_size # Считаем качество модели с таким размером батча.

)

from transformers import BertForSequenceClassification, AdamW, BertConfig



# Загружаем BertForSequenceClassification. Это предобученная модель BERT с одиночным полносвязным слоем для классификации

model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased", # Используем 12-слойную модель BERT, со словарем без регистра.

    num_labels = 2, # Количество выходных слоёв – 2 для бинарной классификации. Можно увеличить для мультиклассовой классификации.

    output_attentions = False, # Будет ли модель возвращать веса для attention-слоёв. В нашем случае нет.

    output_hidden_states = False, # Будет ли модель возвращать состояние всех скрытых слоёв. В нашем случае нет.

)



# Здесь мы говорим PyTorch что хотим тренировать модель на GPU.

if torch.cuda.is_available():

    model.cuda()



# Получаем все параметры модели как список кортежей и выводим сводную информацию по модели.

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



print('\n==== First Transformer ====\n')

for p in params[5:21]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



print('\n==== Output Layer ====\n')

for p in params[-4:]:

    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

optimizer = AdamW(model.parameters(),

    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5

    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.

)



from transformers import get_linear_schedule_with_warmup



# Количество эпох для тренировки. Авторы BERT рекомендуют от 2 до 4.

# Мы выбираем 4, но увидим позже, что это приводит к оверфиту на тренировочные данные.

epochs = 1



# Общее число шагов тренировки равно [количество батчей] x [число эпох].

total_steps = len(train_dataloader) * epochs



# Создаем планировщик learning rate (LR). LR будет плавно уменьшаться в процессе тренировки

scheduler = get_linear_schedule_with_warmup(optimizer,

                                            num_warmup_steps = 0, # Default value in run_glue.py

                                            num_training_steps = total_steps)

import numpy as np



# Функция для расчёта точности. Сравниваются предсказания и реальная разметка к данным

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)





import time

import datetime

import random 



# На вход время в секундах и возвращается строка в формате hh:mm:ss

def format_time(elapsed):

    # Округляем до ближайшей секунды.

    elapsed_rounded = int(round((elapsed)))



    # Форматируем как hh:mm:ss

    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_step(device, model, train_dataloader, optimizer, scheduler):

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    t0 = time.time()

    total_train_loss = 0

    # Переводим модель в режим тренировки.

    model.train()



    # Для каждого батча из тренировочных данных...

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))



        # Извлекаем все компоненты из полученного батча

        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # Очищаем все ранее посчитанные градиенты (это важно)

        model.zero_grad()

        # Выполняем прямой проход по данным

        loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # Накапливаем тренировочную функцию потерь по всем батчам

        total_train_loss += loss.item()

        # Выполняем обратное распространение ошибки что бы посчитать градиенты.

        loss.backward()

        # Ограничиваем максимальный размер градиента до 1.0. Это позволяет избежать проблемы "exploding gradients".

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Обновляем параметры модели используя рассчитанные градиенты с помощью выбранного оптимизатора и текущего learning rate.

        optimizer.step()

        # Обновляем learning rate.

        scheduler.step()



    # Считаем среднее значение функции потерь по всем батчам.

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Сохраняем время тренировки одной эпохи.

    training_time = format_time(time.time() - t0)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(training_time))

    return avg_train_loss, training_time
def validation_step(device, model, validation_dataloader):

    print("Running Validation...")

    t0 = time.time()

    # Переводим модель в режим evaluation – некоторые слои, например dropout ведут себя по другому.

    model.eval()



    # Переменные для подсчёта функции потерь и точности

    total_eval_accuracy = 0

    total_eval_loss = 0

    # Прогоняем все данные из валидации

    for batch in validation_dataloader:

        # Извлекаем все компоненты из полученного батча.

        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)



        # Говорим pytorch что нам не нужен вычислительный граф для подсчёта градиентов (всё будет работать намного быстрее)

        with torch.no_grad():

            # Прямой проход по нейронной сети и получение выходных значений.

            (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)



        # Накапливаем значение функции потерь для валидации.

        total_eval_loss += loss.item()



        # Переносим значения с GPU на CPU

        logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()



        # Считаем точность для отдельного батча с текстами и накапливаем значения.

        total_eval_accuracy += flat_accuracy(logits, label_ids)



    # Выводим точность для всех валидационных данных.

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))



    # Считаем среднюю функцию потерь для всех батчей.

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Измеряем как долго считалась валидация.

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    print("  Validation took: {:}".format(validation_time))

    return avg_val_loss, avg_val_accuracy, validation_time

# В этой переменной сохраним всякую статистику по тренировке: точность, функцию цены (потерь) и время выполнения.

training_stats = []

# Переменная что бы измерить время всей тренировки.

total_t0 = time.time()



# Для каждой эпохи...

for epoch_i in range(0, epochs):

    # Запустить одну эпоху тренировки (следующий слайд) 

    avg_train_loss, training_time = train_step(device, model, train_dataloader, optimizer, scheduler)

    # Запустить валидацию что бы проверить качество модели на данном этапе (следующий слайд)

    avg_val_loss, avg_val_accuracy, validation_time = validation_step(device, model, validation_dataloader)



    # Сохраняем статистику тренировки на данной эпохе.

    training_stats.append(

        {

            'Epoch': epoch_i + 1,

            'Training Loss': avg_train_loss,

            'Validation Loss': avg_val_loss,

            'Validation Accur.': avg_val_accuracy,

            'Training Time': training_time,

            'Validation Time': validation_time

        }

    )



print("Training complete! Total training took {:} (hh:mm:ss)".format(format_time(time.time() - total_t0)))

import os



# Задаем выходную директорию

output_dir = './model_save/'

# Если она не существует создаем её

if not os.path.exists(output_dir):

    os.makedirs(output_dir)



print("Saving model to %s" % output_dir)



# Сохраняем натренированную модель и её токенайзер используя `save_pretrained()`.

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

model_to_save.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)
from transformers import BertTokenizer, BertForSequenceClassification

# Загружаем натренированную модель и её словарь

model = BertForSequenceClassification.from_pretrained(output_dir)

tokenizer = BertTokenizer.from_pretrained(output_dir)



# Отправляем модель на GPU.

if torch.cuda.is_available():

    model.to(device)