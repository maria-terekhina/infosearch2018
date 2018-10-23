import re


#удаляем из текста символы переноса строки
#убираем пробелы перед сокращенными названиями населенных пуктов
def no_spaces(text):
    
    processed_text = text.replace('\n', ' ').replace('\n\n', ' ').replace('ул. ', 'ул.').replace('г. ', 'г.').replace('гор. ', 'гор.').replace('с. ', 'с.')
    return processed_text


#убираем пробел после инициалов перед фамилией
def clear_abbrs(processed_text):
    initials = re.compile(r'[А-Я]{1}\.([А-Я]{1}\.)? [А-Я][а-яё]+')
    counter = len(initials.findall(processed_text))

    for s in range(counter):
        get_abbrs = initials.search(processed_text)
        i = get_abbrs.span()[0] + 4
        processed_text = processed_text[:i] + processed_text[i+1:]
    return processed_text


#делим текст на предложения при помощи регулярного выражения
def split_text(processed_text):
    
    text_splitted = re.split(r'(\. +[А-Я]{1} *[а-яё]+)', processed_text)
    last_word = re.compile(r'[А-Я]{1} *[а-яё]+')
    normal_sentences = [text_splitted[0] + '.']

    for i in range(1, len(text_splitted), 2):
        if i + 1 <= len(text_splitted)-1:
            beginning = last_word.findall(text_splitted[i])[0]
            normal_sentences.append(beginning + text_splitted[i+1] + '.')
        elif i == len(text_splitted)-1:
            beginning = last_word.findall(text_splitted[i])[0]
            normal_sentences.append(beginning)
    return normal_sentences


def get_sentences(text):
    text = no_spaces(text)
    text = clear_abbrs(text)
    sentences = split_text(text)
    return sentences


#делим текст на куски по n предложений
#(функция принимает на вход список из предложений-строк, полученный на предыдущем шаге)
def split_paragraph(list_of_sentences, n):

    l = len(list_of_sentences)

    n_chunks = []
    chunk = ''

    for i in range(0, l, n):
        for j in range(n):
            if i+j < l:
                chunk += list_of_sentences[i+j] + ' '
            else:
                continue
        n_chunks.append(chunk)
        chunk = ''
    return n_chunks


def splitter(text, n):
    normal_sentences = get_sentences(text)
    splitted_sentences = split_paragraph(get_sentences(text), n) 
    return splitted_sentences
    
    
    
    