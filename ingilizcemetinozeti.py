import math
import nltk
import string
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from operator import itemgetter

from transformers import PegasusTokenizer, PegasusForConditionalGeneration


model_name = "google/pegasus-xsum"

pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
example_text = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. 
Learning can be supervised semi-supervised or unsupervised.
 Deep-learning architectures such as deep neural networks deep belief networks deep reinforcement learning recurrent neural networks and convolutional 
 neural networks have been applied to fields including computer vision speech recognition natural language processin machine translation 
 bioinformatics drug design medical image analysis material inspection and board game programs where they have produced results comparable 
 to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed 
 communication nodes in biological systems. ANNs have various differences from biological brains. Specifically neural networks tend to be static and symbolic
 while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective deep in deep learning refers to the use of multiple layers in the network. 
 Early work showed that a linear perceptron cannot be a universal classifier but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
   Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size which permits practical application and optimized implementation
   while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed
     connectionist models for the sake of efficiency trainability and understandability whence the structured part."""

# PEGASUS modelini tanımladık
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Token oluşturduk
tokens = pegasus_tokenizer(example_text, truncation=True, padding="longest", return_tensors="pt")

# metin verisinin özetini oluşturmak için pegasus_model.generate yöntemi çağrılır. en fazla 5000 yeni token içeren bir özet oluşturulur. 
#**tokens ifadesi, bu generate yönteminin belirli parametreleri alabilmesini sağlar.
encoded_summary = pegasus_model.generate(**tokens, max_new_tokens=5000)

# pegasus_tokenizer.decode yöntemi, tokenleri metne dönüştürmek için kullanılır.
# encoded_summary[0] ifadesi, encoded_summary değişkeninin ilk öğesini temsil eder. Bu, özetin kodlanmış temsilcisini içeren bir liste
# pegasus_tokenizer.decode decoded_summary değişkenine metin bir özet döndürür
# skip_special_tokens=True metindeki özel belirteçleri atlar ve sadece gerçek metin içeriğini döndürür. Bu, özetin daha okunabilir ve anlaşılır bir formda sunulmasını sağlar.
decoded_summary = pegasus_tokenizer.decode(
      encoded_summary[0],
      skip_special_tokens=True
)

print(decoded_summary)

# İngilizce stopwords listesini yükleyin
stop_words = set(stopwords.words('english'))

# Kelimeleri tokenize etme ve küçük harfe dönüştürme
words = [word.lower() for word in word_tokenize(example_text)]

# Porter Stemmer kullanarak metin verilerindeki kelimelerin köklerini (stem) bulmak için
# temmed_words = [stemmer.stem(word) for word in words] satırı, words adlı bir liste içerisindeki kelimelerin köklerini bulur. 
# For döngüsü ile listenin her bir kelimesi için stemmer.stem(word) ifadesi kullanılarak kök bulunur ve köklenmiş kelimeler stemmed_words listesine eklenir.
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]

def preprocess_text(text):
    # Cümlelere ayırma
    sentences = tokenize.sent_tokenize(text)

    # Metin ön işleme adımlarını uygulama
    # preprocessed_sentences adında boş bir liste oluşturulur.
    preprocessed_sentences = [] 
    for sentence in sentences:
        # Tokenize etme
        words = word_tokenize(sentence)

        # Küçük harflere dönüştürme
        words = [word.lower() for word in words]

        # Noktalama işaretlerini çıkartma
        # string.punctuation içindeki noktalama işaretlerini çıkartmak
        words = [word for word in words if word not in string.punctuation]

        # Stop kelimeleri çıkartma
        words = [word for word in words if word not in stop_words]

        # Stemming yapma
        words = [stemmer.stem(word) for word in words]

        # Her bir cümle için işlemler tamamlandıktan sonra, temizlenmiş kelimeler preprocessed_sentences listesine eklenir.
        preprocessed_sentences.append(words)

    # total_sentences değişkenine, ön işleme adımlarından geçirilmiş cümleleri içeren preprocessed_sentences listesi atanır.
    total_sentences = preprocessed_sentences
    # len(total_sentences) ifadesi kullanılarak total_sentences içindeki cümlelerin toplam sayısı total_sent_len değişkenine atanır.
    total_sent_len = len(total_sentences)

    return preprocessed_sentences, total_sentences, total_sent_len

#  belirli bir kelimenin bir dizi cümle içinde geçip geçmediğini kontrol eder.
def check_sent(word, sentences):
    # all([w in x for w in word]) ifadesi, word içindeki her bir kelimenin, bir cümle içinde geçip geçmediğini kontrol eder. 
    # Burada w in x ifadesi, w kelimesinin x cümlesinde bulunup bulunmadığını kontrol eder
    final = [all([w in x for w in word]) for x in sentences]

    #final listesi, sentences listesindeki her bir cümle için bu değerlendirmeyi yapar ve sonuçları içerir. 
    # Her bir değer, word içindeki tüm kelimeleri içeren bir cümle için True olur, aksi takdirde False olur.
    # sent_len adında yeni bir liste oluşturulur. Bu liste, final listesinde True olan tüm cümleleri içerir. 
    # len(sent_len) ifadesi kullanılarak, sent_len listesindeki cümlelerin toplam sayısı döndürülür.
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return len(sent_len)

# Önişleme adımlarını uygulama
preprocessed_sentences, total_sentences, total_sent_len = preprocess_text(example_text)

# cleaned_words adında boş bir liste oluşturulur. Bu liste, temizlenmiş kelimeleri depolamak için kullanılacak.
# For döngüsü, words adı verilen bir liste içindeki her bir kelime için döner.
# word.replace('.', '') ifadesi, word içindeki noktalama işaretlerini (.) çıkartır.
# if word not in stop_words: ifadesi, word'ün stop kelimeler listesinde bulunmadığını kontrol eder. 
# stemmer.stem(word) ifadesi kullanılarak, kelimenin kökünü (stem) bulmak için 
# Temizlenmiş kelime cleaned_words listesine eklenir.
# For döngüsü tamamlandığında, cleaned_words listesi, temizlenmiş ve köklenmiş kelimeleri içerir.
cleaned_words = []
for word in words:
    word = word.replace('.', '')
    if word not in stop_words:
        cleaned_words.append(stemmer.stem(word))

# TF hesaplama
# For döngüsü tamamlandığında, tf_score sözlüğü, temizlenmiş ve köklenmiş kelimelerin frekanslarını içerir. 
# tf_score adında boş bir sözlük (dictionary) oluşturulur. Bu sözlük, kelimenin frekansını ve kelimenin frekansını temsil eden değerleri depolamak için kullanılacak.
# For döngüsü, cleaned_words adı verilen bir liste içindeki her bir kelime için döner.
# if word in tf_score: ifadesi, word'ün tf_score sözlüğünde zaten bir anahtar olarak bulunup bulunmadığını kontrol eder.
# Eğer kelime tf_score sözlüğünde bir anahtar olarak bulunmuyorsa, tf_score[word] = 1 ifadesi kullanılarak, word'ün frekans değeri 1 olarak atanır. 
tf_score = {}
for word in cleaned_words:
    if word in tf_score:
        tf_score[word] += 1
    else:
        tf_score[word] = 1

# total_word_length adında bir değişken oluşturulur ve bu değişkene temizlenmiş ve köklenmiş kelimelerin toplam sayısı atanır. 
# len(cleaned_words) ifadesi kullanılarak bu hesaplama gerçekleştirilir.
total_word_length = len(cleaned_words)
# tf_score sözlüğündeki her bir değer döngüye alır
# her bir anahtar-değer çifti için  xy oluşturur 
# y değerini temizlenmiş ve köklenmiş kelimelerin toplam sayısına (total_word_length) böler.
# her bir kelimenin metindeki sıklığını elde etmek için 
tf_score.update((x, y / total_word_length) for x, y in tf_score.items())

# IDF hesaplama
idf_score = {}
for word in cleaned_words:
    # Eğer word zaten idf_score sözlüğünde bulunuyorsa, idf_score[word] = check_sent(word, total_sentences) ifadesi kullanılarak,
    #  word'ün IDF skoru check_sent(word, total_sentences) değeri ile güncellenir. 
    if word in idf_score:
        idf_score[word] = check_sent(word, total_sentences)
    #e ğer word idf_score sözlüğünde bulunmuyorsa, idf_score[word] = 1 ifadesi kullanılarak, word'ün IDF skoru 1 olarak atanır.
    #  Bu durumda, kelimenin yalnızca bir belgede geçtiği varsayılır.
    else:
        idf_score[word] = 1

# IDF skorlarını hesaplama
# her bir terim ve bu terimin belgedeki görülme sıklığı (y) için bir döngü yapılır.
# total_sent_len, belgenin toplam cümle uzunluğunu temsil eder. y, terimin belgedeki görülme sıklığını temsil eder. y + 1 ifadesi, eğer bir terim belgede 
# hiç görülmezse sıfıra bölme hatasını önlemek için kullanılır. 
# Bu şekilde, her terimin IDF skoru güncellenir ve idf_score sözlüğüne atanır.
idf_score.update((x, math.log(total_sent_len / (y + 1))) for x, y in idf_score.items())

# TF-IDF skorlarını hesaplama
# idf_score.get(key, 0) ifadesi, idf_score sözlüğünden terimin IDF skorunu alır. Eğer terim idf_score sözlüğünde bulunamazsa, varsayılan olarak 0 değeri kullanılır. 
# Terim frekansı (TF) skoru tf_score[key] değeriyle elde edilir. Bu, terimin belgedeki görülme sıklığını ifade eder.
tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

# En yüksek 5 anahtar kelimeyi alma
# sorted() fonksiyonu, anahtar-değer çiftlerini değerlerine göre sıralamak için kullanılır. 
# key=itemgetter(1) ifadesi, çiftlerin ikinci öğelerine (değerler) göre sıralama yapılacağını belirtir. reverse=True parametresi, sıralamanın büyükten küçüğe olmasını sağlar.
# Sonuç olarak elde edilen sıralı çiftler, dict() fonksiyonuyla tekrar bir sözlüğe dönüştürülür.
# Oluşturulan sözlük, result değişkenine atanır.
# result sözlüğü, en yüksek N anahtar kelimeyi ve bu anahtar kelimelerin değerlerini içerir.
# Son olarak, result sözlüğü fonksiyondan döndürülür.
def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
    return result

print(get_top_n(tf_idf_score, 5))