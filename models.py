'''
运行环境：
    python: 3.9.25
    cudatoolkit: 11.3.1
    cudnn: 8.2.1
    tensorflow: 2.10.0
    transformers: 4.20.0
    numpy: 1.26.4
    其余库不存在兼容性问题
在非兼容环境下运行时tensorflow可能无法正常调用GPU
'''


import pickle
import re
import pandas as pd
import os
import time
import joblib
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import AdamWeightDecay
warnings.filterwarnings('ignore')


current_dir = os.path.dirname(os.path.abspath(__file__))
current_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
result_path = os.path.join(current_dir, f'Training Results\{current_time}.txt')  # 训练结果日志
os.makedirs(os.path.dirname(result_path), exist_ok=True)
f = open(result_path, 'w', encoding='utf-8')


'''检查GPU是否可用'''
print("\n" + '='*50)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow 识别到 {len(gpus)} 个 GPU:")
    f.write(f"TensorFlow 识别到 {len(gpus)} 个 GPU:\n")
    for i, gpu in enumerate(gpus):
        try:
            # 尝试获取更详细的GPU信息
            details = tf.config.experimental.get_device_details(gpu)
            gpu_name = details['device_name']
            print(f"GPU {i}: {gpu.name} - {gpu_name}")
            f.write(f"GPU {i}: {gpu.name} - {gpu_name}")
        except:
            # 如果获取详细信息失败，只打印基本名称
            print(f"GPU {i}: {gpu.name}")
            f.write(f"GPU {i}: {gpu.name}")
else:
    print("未检测到GPU，将使用CPU运行")
    f.write("未检测到GPU，将使用CPU运行")
print("="*50)


'''
文本预处理模块：
'''


# 用于初始化训练集和测试集文件，并对其去除干扰内容和进行分词（即预处理）
class init_reviews:
    def __init__(self, positive_path, negative_path, test_path, name='cn'):
        # 初始化训练集
        self.train_df = self.parse_train_reviews(positive_path, negative_path)
        print(f"{name}训练集大小: {len(self.train_df)}")
        f.write(f"\n{name}训练集大小: {len(self.train_df)}")

        # 初始化测试集
        self.test_df = self.parse_test_reviews(test_path)
        print(f"{name}测试集大小: {len(self.test_df)}")
        f.write(f"\n{name}测试集大小: {len(self.test_df)}")

    def get(self):
        return self.train_df['review'], self.train_df['label'], self.test_df['review'], self.test_df['label']
    
    def clean(self, text):
        '''用于去除文本内的干扰内容'''
        cleaned_text = re.sub(r'\n+', ' ', text).strip()    # 清除换行
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)    # 清除空格
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,!?;:，。！？；："\'\-()（）【】《》]', '', cleaned_text) # 清除特殊字符
        return cleaned_text

    def tokenize_chinese(self, text):
        '''用于对中文进行精确分词，并过滤停用词'''
        words = jieba.lcut(text, cut_all=False) # 精确模式分词
        
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            '着', '没有', '看', '好', '自己', '这', '那', '啊', '哦', '嗯',
            '吧', '呢', '吗', '哈', '呀', '啦', '哇', '噢', '喔', '唉'
            }
        words = [word for word in words if word not in stopwords and len(word) > 1] # 去除停用词
        
        return ' '.join(words)
    

    def parse_train_reviews(self, positive_sample, negative_sample):
        '''对训练集进行预处理'''
        reviews = []
        pattern = r'<review id="(\d+)">\s*(.*?)\s*</review>'    # 匹配格式: <review id="数字">文本内容</review>
    
        with open(positive_sample, 'r', encoding='utf-8') as fp: positive_text = fp.read()
        with open(negative_sample, 'r', encoding='utf-8') as fn: negative_text = fn.read()
        
        for match in re.finditer(pattern, positive_text, re.DOTALL):
            review_id = match.group(1)  # 提取ID
            review_text = match.group(2)  # 提取评论文本
            
            review_text = self.clean(review_text)
            review_text = self.tokenize_chinese(review_text)
            
            reviews.append({
                'review_id': review_id,
                'review': review_text,
                'label': 1
            })

        for match in re.finditer(pattern, negative_text, re.DOTALL):
            review_id = match.group(1)  # 提取ID
            review_text = match.group(2)  # 提取评论文本
            
            review_text = self.clean(review_text)
            review_text = self.tokenize_chinese(review_text)
            
            reviews.append({
                'review_id': review_id,
                'review': review_text,
                'label': 0
            })

        # 转换为DataFrame
        df = pd.DataFrame(reviews)
        df['label'] = df['label'].astype(int)
        return df
    
    def parse_test_reviews(self, test_sample):
        '''对测试集进行预处理'''
        reviews = []
        pattern = r'<review\s+id="(\d+)"\s+label="(\d+)"\s*>\s*(.*?)\s*</review>'   # 匹配格式: <review id="数字" label="数字">文本内容</review>

        with open(test_sample, 'r', encoding='utf-8', errors='ignore') as f: test_text = f.read()
        # 使用errors='ignore'是由于发现test.label.en.txt文件有不可读的版权符号®: byte 0xae
        
        for match in re.finditer(pattern, test_text, re.DOTALL):
            review_id = match.group(1)  # 提取ID
            review_label = match.group(2) # 提取label
            review_text = match.group(3)  # 提取评论文本
            
            review_text = self.clean(review_text)
            review_text = self.tokenize_chinese(review_text)
            
            reviews.append({
                'review_id': review_id,
                'review': review_text,
                'label': review_label
            })

        # 转换为DataFrame
        df = pd.DataFrame(reviews)
        df['label'] = df['label'].astype(int)
        return df


'''
以下为传统模型部分:
'''


# 创建用于存放各语言模型的文件夹
models_dir = os.path.join(current_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, 'Chinese'), exist_ok=True)
os.makedirs(os.path.join(models_dir, 'English'), exist_ok=True)
os.makedirs(os.path.join(models_dir, 'Bilingual'), exist_ok=True)


class tradition_models:
    def __init__(self, X_train, y_train, X_test, y_test,
                name='chinese'   # 用于区分中文、英文和双语模型
                ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.name = name

        self.TFIDF_dir = os.path.join(models_dir, name, 'TFIDF.pkl')   # TF-IDF模型保存路径
        self.NaiveBayes_dir = os.path.join(models_dir, name, 'Naive Bayes.pkl')    # Naive Bayes模型保存路径
        self.SVM_dir = os.path.join(models_dir, name, 'SVM.pkl')   # SVM模型保存路径
        self.RandomForest_dir = os.path.join(models_dir, name, 'Random Forest.pkl')    # Random Forest模型保存路径

        # 加载时立即执行TFIDF处理
        self.TF_IDF()

    def TF_IDF(self):
        '''TF-IDF处理'''
        print("\n"+"="*50)
        try:
            # 尝试加载已保存的TF-IDF模型
            tfidf = joblib.load(self.TFIDF_dir)

            # 加载模型 已保存的tfidf无需再次fit
            self.X_train_tfidf = tfidf.transform(self.X_train)
            self.X_test_tfidf = tfidf.transform(self.X_test)
            print("TF-IDF向量化器加载成功")
        except  (FileNotFoundError, pickle.UnpicklingError):    # pickle.UnpicklingError防止模型文件损坏
            print("未找到TFIDF.pkl，执行TF-IDF向量化器中...")
            
            # TF-IDF特征化（只在训练集上拟合）
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=['english'] if self.name != 'Chinese' else None  # 根据语言设置使用英文停用词
            )
            
            # 对训练集进行fit_transform 对测试集进行transform（使用训练集的词汇表）
            self.X_train_tfidf = tfidf.fit_transform(self.X_train)
            self.X_test_tfidf = tfidf.transform(self.X_test)
            
            # 保存模型
            joblib.dump(tfidf, self.TFIDF_dir)
            print("TF-IDF向量化器处理完毕，已保存至", self.TFIDF_dir)
        print("="*50)

    def NaiveBayes(self):
        self.train(MultinomialNB(), 'Naive Bayes')

    def SVM(self):
        self.train(SVC(kernel='linear', probability=True, random_state=42), 'SVM')

    def RandomForest(self):
        self.train(RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest')

    def train(self, model_func, name):
        '''由于三个传统模型训练步骤重复，故整合为一个函数'''
        
        # 索引三个传统模型的路径
        model_dir = {
            'Naive Bayes': self.NaiveBayes_dir,
            'SVM': self.SVM_dir,
            'Random Forest': self.RandomForest_dir
        }
        
        print("\n"+"="*50)
        print(f"{name}_{self.name}:")
        try:
            model = joblib.load(filename=model_dir[name])    # 尝试加载已保存的模型文件
            y_pred = model.predict(self.X_test_tfidf)    # 转化为预测模型
            print("模型加载成功")
        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"未找到已保存的{name}_{self.name}.pkl，训练{name}模型中...")
            model = model_func  # model_func用于传入训练模型时执行的函数
            # 训练模型并预测
            model.fit(self.X_train_tfidf, self.y_train)
            y_pred = model.predict(self.X_test_tfidf)

            # 保存模型
            joblib.dump(model, model_dir[name])
            print(f"训练完毕，已保存至{model_dir[name]}")
        self.evaluate(y_pred, name) # 对模型进行评估
    
    def evaluate(self, y_pred, name):
        print("="*50)
        print(f"{name}_{self.name}分类报告:")
        report = classification_report(self.y_test, y_pred, zero_division=0)
        print(report)
        f.write(f"\n\n{name}_{self.name}分类报告:\n{report}")


'''
以下为CNN模型部分:
'''


class CNN:
    def __init__(self, X_train, y_train, X_test, y_test,
                max_len=200,    # 最大序列长度
                max_words=50000,    # 词汇表大小
                embedding_dim=100,   # 词向量维度
                name='chinese'   # 用于区分中文、英文和双语模型
                ):
                    
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_len = max_len
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.name = name
                    
        # 默认设置tokenizer和model为空
        self.tokenizer = None
        self.model = None
        
        self.Tokenizer_dir = os.path.join(models_dir, name, 'Tokenizer.pkl')   # Tokenizer保存路径
        self.CNN_dir = os.path.join(models_dir, name, 'CNN.h5')    # CNN模型保存路径

        # 预处理数据
        self.X_train_pad, self.X_test_pad, self.y_train_np, self.y_test_np = self.preprocess_data()
    
    def preprocess_data(self):
        '''预处理数据：分词、序列化、填充'''
        print("\n"+"="*50)
        print(f"CNN_{self.name}:")
        try:
            # 尝试加载tokenizer结果
            self.tokenizer = joblib.load(self.Tokenizer_dir)
            print(f"Tokenizer_{self.name}加载成功")
        except FileNotFoundError:
            print(f"未找到Tokenizer_{self.name}.pkl，执行Tokenizer_{self.name}中...")
            
            # 执行预处理
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(self.X_train)

            # 保存tokenizer结果
            joblib.dump(self.tokenizer, self.Tokenizer_dir)
            print(f"Tokenizer_{self.name}处理完毕，已保存至", self.Tokenizer_dir)
        
        # 将文本转换为序列
        X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        
        # 填充序列到相同长度
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post')
        
        # 将标签转换为numpy数组
        y_train_np = np.array(self.y_train)
        y_test_np = np.array(self.y_test)
        
        return X_train_pad, X_test_pad, y_train_np, y_test_np
    
    def build_model(self):
        '''构建CNN模型'''
        model = Sequential([
            # 嵌入层
            Embedding(input_dim=self.max_words, 
                    output_dim=self.embedding_dim, 
                    input_length=self.max_len),
            
            # 第一个卷积层
            Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # 第二个卷积层
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # 第三个卷积层
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # 展平层
            Flatten(),
            
            # 全连接层
            Dense(64, activation='relu'),
            Dropout(0.5),
            
            # 输出层（二分类）
            Dense(1, activation='sigmoid')
        ])
        
        # 编译模型
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        return model
    
    def train(self, epochs=10, batch_size=32, validation_split=0.1):
        '''训练CNN模型'''
        try:
            # 尝试加载已保存的模型
            self.model = load_model(self.CNN_dir)
            print(f"CNN_{self.name}模型加载成功")
        except (FileNotFoundError, OSError):
            print(f"未找到CNN_{self.name}.h5，训练CNN_{self.name}模型中...")
            
            # 构建模型
            self.model = self.build_model()

            # 打印模型结构
            print("CNN模型结构:")
            self.model.summary()
            
            # 设置早停和模型检查点
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.CNN_dir,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # 训练模型
            print(f"开始训练CNN_{self.name}模型...")
            self.model.fit(
                self.X_train_pad, self.y_train_np,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            print(f"CNN_{self.name}模型训练完毕，已保存至", self.CNN_dir)
        print("="*50)
        
        # 训练完毕自动生成分类报告
        self.evaluate()
    
    def evaluate(self):
        '''评估CNN模型'''
        # 预测
        y_pred_prob = self.model.predict(self.X_test_pad, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # 生成分类报告
        print(f"\nCNN_{self.name}分类报告:")
        report = classification_report(self.y_test_np, y_pred, zero_division=0)
        print(report)
        f.write(f"\n\nCNN_{self.name}分类报告:\n{report}")


'''
以下为BERT模型部分:
'''


class BERT:
    def __init__(self, X_train, y_train, X_test, y_test,
                max_len=128,          # BERT最大序列长度（通常128或256）
                model_name='bert-base-chinese',  # 预训练模型名称
                num_labels=2,          # 分类类别数
                name='Chinese'   # 用于区分中文、英文和双语模型
                ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_len = max_len
        self.model_name = model_name
        self.num_labels = num_labels
        self.name = name

        # 默认设置tokenizer和model为空
        self.tokenizer = None
        self.model = None

        self.BERT_tokenizer_dir = os.path.join(models_dir, name, 'BERT', 'BERT_tokenizer')   # BERT tokenizer保存目录
        self.BERT_model_dir = os.path.join(models_dir, name, 'BERT', 'BERT_model')          # BERT模型保存目录
        
        # 初始化tokenizer和模型
        self.init_tokenizer()
        
        # 预处理数据
        self.X_train_enc, self.X_test_enc, self.y_train_np, self.y_test_np = self.preprocess_data()

    def init_tokenizer(self):
        '''初始化BERT tokenizer'''
        print("\n" + "="*50)
        print(f"BERT_{self.name}模型初始化:")
        try:
            # 尝试加载已保存的tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.BERT_tokenizer_dir)
            print(f"BERT tokenizer_{self.name}加载成功")
        except:
            # 加载预训练tokenizer
            print(f"加载预训练tokenizer_{self.name}: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            
            # 保存tokenizer
            self.tokenizer.save_pretrained(self.BERT_tokenizer_dir)
            print(f"BERT tokenizer_{self.name}已保存至: {self.BERT_tokenizer_dir}")
        print("="*50)
    
    def encode_texts(self, texts):
        '''将文本编码为BERT输入格式'''
        encodings = self.tokenizer(
            texts.tolist() if hasattr(texts, 'tolist') else texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']
    
    def preprocess_data(self):
        '''预处理数据：BERT编码'''
        print("数据预处理中...")
        
        # 将标签转换为numpy数组
        y_train_np = np.array(self.y_train)
        y_test_np = np.array(self.y_test)
        
        # 编码训练集和测试集
        X_train_ids, X_train_mask = self.encode_texts(self.X_train)
        X_test_ids, X_test_mask = self.encode_texts(self.X_test)
        
        # 组合编码结果
        X_train_enc = {
            'input_ids': X_train_ids,
            'attention_mask': X_train_mask
        }
        
        X_test_enc = {
            'input_ids': X_test_ids,
            'attention_mask': X_test_mask
        }
        
        print(f"序列长度: {self.max_len}")
        
        return X_train_enc, X_test_enc, y_train_np, y_test_np
    
    def build_model(self):
        '''构建BERT分类模型 - 直接使用transformers模型'''
        print(f"构建BERT_{self.name}模型中...")
        
        # 加载transformers模型
        model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        
        # 配置优化器
        optimizer = AdamWeightDecay(
            learning_rate=2e-5,
            weight_decay_rate=0.01,
            epsilon=1e-8,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )
        
        # 编译模型
        if self.num_labels == 1:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = ['accuracy']
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def train(self, epochs=3, batch_size=16, validation_split=0.1):
        '''训练（微调）BERT模型'''
        print("\n" + "="*50)
        print(f"BERT_{self.name}模型训练:")
        
        # 创建目录
        os.makedirs(self.BERT_model_dir, exist_ok=True)
        
        try:
            # 尝试加载模型
            self.model = TFBertForSequenceClassification.from_pretrained(self.BERT_model_dir)
            self._compile_model()   # 重新编译模型（因为保存时可能没有优化器状态）
            
            print(f"BERT_{self.name}模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("重新构建模型...")
            print(f"未找到BERT_{self.name}模型，开始微调训练:{e}")
            
            # 构建模型
            self.model = self.build_model()
            
            # 打印模型结构
            print("BERT模型结构:")
            try:
                self.model.summary()
            except:
                print("模型summary不可用")
            
            # 设置简单的回调函数
            checkpoint_path = os.path.join(self.BERT_model_dir, 'checkpoint.weights.h5')
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            ]
            
            # 准备训练数据
            train_inputs = [self.X_train_enc['input_ids'], self.X_train_enc['attention_mask']]
            
            # 训练模型
            print(f"开始微调BERT_{self.name}模型...")
            try:
                self.model.fit(
                    x=train_inputs,
                    y=self.y_train_np,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
                # 保存模型
                print("保存模型...")
                self.model.save_pretrained(self.BERT_model_dir)
                self.tokenizer.save_pretrained(self.BERT_model_dir)
                print(f"BERT_{self.name}模型已保存至: {self.BERT_model_dir}")
            except Exception as e:
                print(f"训练或保存过程中出错: {e}")
                import traceback
                traceback.print_exc()
        
        print("="*50)

        # 训练完自动生成分类报告
        return self.evaluate()
    
    def _compile_model(self):
       '''编译模型（用于加载后重新编译）'''
        optimizer = AdamWeightDecay(
            learning_rate=2e-5,
            weight_decay_rate=0.01,
            epsilon=1e-8,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )
        
        if self.num_labels == 1:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
    
    def evaluate(self):
        '''评估BERT模型'''
        print(f"\nBERT_{self.name}模型评估:")
                
        # 准备测试数据
        test_inputs = [self.X_test_enc['input_ids'], self.X_test_enc['attention_mask']]
        
        try:
            # 预测
            predictions = self.model.predict(
                test_inputs,
                batch_size=16,
                verbose=0
            )
            
            # 获取logits
            if hasattr(predictions, 'logits'):
                y_pred_logits = predictions.logits
            elif isinstance(predictions, dict) and 'logits' in predictions:
                y_pred_logits = predictions['logits']
            else:
                y_pred_logits = predictions
            
            # 处理预测结果
            y_pred = self._process_predictions(y_pred_logits)
            
            # 生成分类报告
            report = classification_report(self.y_test_np, y_pred, zero_division=0)
            print(report)
            f.write(f"\n\nBERT_{self.name}分类报告:\n{report}")
            
        except Exception as e:
            print(f"评估过程中出错: {e}")
            import traceback
            traceback.print_exc()

    def _process_predictions(self, y_pred_logits):
        '''统一处理预测结果'''
        y_pred_logits = np.array(y_pred_logits)
        
        # 如果是二分类
        if self.num_labels == 1:
            # 从logits转换为概率
            y_pred_prob = 1 / (1 + np.exp(-y_pred_logits.flatten()))
            return (y_pred_prob > 0.5).astype(int)
        # 如果是多分类
        else:
            return np.argmax(y_pred_logits, axis=1)
        


'''
以下部分为用于对输入预测的预测器：
'''


class TextPredictor:
    '''文本情感预测器'''
    def __init__(self, language='Chinese'):
        self.language = language
        self.models_dir = models_dir
        self.language_dir = os.path.join(models_dir, language)
        
        # 加载所有必要的预处理工具和模型
        self.load_models()
    
    def load_models(self):
        '''加载所有模型和预处理工具'''
        print(f"\n加载{self.language}模型...")
        
        # 加载TF-IDF和传统模型
        try:
            self.tfidf = joblib.load(os.path.join(self.language_dir, 'TFIDF.pkl'))
            print("TF-IDF加载成功")
        except:
            raise FileNotFoundError("TF-IDF未找到")
        
        # 加载传统机器学习模型
        self.traditional_models = {}
        traditional_model_files = {
            'Naive Bayes': 'Naive Bayes.pkl',
            'SVM': 'SVM.pkl',
            'Random Forest': 'Random Forest.pkl'
        }
        
        for model_name, filename in traditional_model_files.items():
            try:
                model_path = os.path.join(self.language_dir, filename)
                self.traditional_models[model_name] = joblib.load(model_path)
                print(f"{model_name}加载成功")
            except:
                raise FileNotFoundError(f"{model_name}加载失败")
        
        # 加载CNN模型
        try:
            self.cnn_tokenizer = joblib.load(os.path.join(self.language_dir, 'Tokenizer.pkl'))
            self.cnn_model = load_model(os.path.join(self.language_dir, 'CNN.h5'))
            print("CNN模型加载成功")
        except:
            raise FileNotFoundError("CNN模型加载失败")
        
        # 加载BERT模型
        try:
            bert_model_dir = os.path.join(self.language_dir, 'BERT', 'BERT_model')
            bert_tokenizer_dir = os.path.join(self.language_dir, 'BERT', 'BERT_tokenizer')
            
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_dir)
            self.bert_model = TFBertForSequenceClassification.from_pretrained(bert_model_dir)
            print("BERT模型加载成功")
        except:
            raise FileNotFoundError("BERT模型加载失败")
        
        print("所有模型加载完成")
    
    def preprocess_text(self, text):
        '''预处理输入文本'''
        # 清洗文本
        cleaned_text = re.sub(r'\n+', ' ', text).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,!?;:，。！？；："\'\-()（）【】《》]', '', cleaned_text)
        
        # 中文分词
        if self.language == 'Chinese':
            words = jieba.lcut(cleaned_text, cut_all=False)
            stopwords = {
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
                '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
                '着', '没有', '看', '好', '自己', '这', '那', '啊', '哦', '嗯',
                '吧', '呢', '吗', '哈', '呀', '啦', '哇', '噢', '喔', '唉'
            }
            words = [word for word in words if word not in stopwords and len(word) > 1]
            processed_text = ' '.join(words)
        else:
            # 英文处理
            processed_text = cleaned_text.lower()
        return processed_text
    
    def predict_traditional(self, text, model_name):
        '''使用传统机器学习模型预测'''
        if model_name not in self.traditional_models or self.tfidf is None:
            return None
            
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # TF-IDF转换
        text_tfidf = self.tfidf.transform([processed_text])
        
        # 预测
        y_pred = self.traditional_models[model_name].predict(text_tfidf)[0]
        result = "正面" if y_pred == 1 else "负面"
        print(f"使用{model_name}_{self.language}模型预测内容：\n    '{text}'\n结果为：{result}\n")
        f.write(f"使用{model_name}_{self.language}模型预测内容：\n    '{text}'\n结果为：{result}\n\n")
        return result
    
    def predict_cnn(self, text):
        '''使用CNN模型预测'''
        if self.cnn_model is None or self.cnn_tokenizer is None:
            return None
            
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 序列化和填充
        sequence = self.cnn_tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=200, padding='post')
        
        # 预测
        prediction = self.cnn_model.predict(padded, verbose=0)[0][0]
        result = "正面" if prediction > 0.5 else "负面"
        print(f"使用CNN_{self.language}模型预测内容：\n    '{text}'\n结果为：{result}\n'")
        f.write(f"使用CNN_{self.language}模型预测内容：\n    '{text}'\n结果为：{result}\n\n")
        return result
    
    def predict_bert(self, text):
        '''使用BERT模型预测'''
        if self.bert_model is None or self.bert_tokenizer is None:
            return None
            
        # BERT编码
        encodings = self.bert_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='tf'
        )
        
        # 预测
        predictions = self.bert_model.predict(
            [encodings['input_ids'], encodings['attention_mask']],
            verbose=0
        )
        
        # 处理预测结果
        if hasattr(predictions, 'logits'):
            logits = predictions.logits[0]
        else:
            logits = predictions[0]
        
        # 获取预测类别
        predicted_class = tf.argmax(logits, axis=-1).numpy()
        result = "正面" if predicted_class == 1 else "负面"
        print(f"使用BERT_{self.language}模型预测内容：\n    '{text}'\n结果为：{result}\n")
        f.write(f"使用BERT_{self.language}模型预测内容：\n    '{text}'\n结果为：{result}\n\n")
        return result
    
    def predict_all(self, text):
        '''使用所有模型进行综合预测'''
        print("-" * 50)
        
        results = {}
        
        # 传统模型预测
        for model_name in ['Naive Bayes', 'SVM', 'Random Forest']:
            result = self.predict_traditional(text, model_name)
            if result is not None:
                results[model_name] = result
        
        # CNN预测
        cnn_result = self.predict_cnn(text)
        if cnn_result is not None:
            results['CNN'] = cnn_result
        
        # BERT预测
        bert_result = self.predict_bert(text)
        if bert_result is not None:
            results['BERT'] = bert_result
        
        print("-" * 50)
        
        # 统计结果
        if results:
            pos_count = sum(1 for v in results.values() if v == "正面")
            neg_count = sum(1 for v in results.values() if v == "负面")
            print(f"\n综合结果: {pos_count}个模型预测为正面, {neg_count}个模型预测为负面")
            # 返回投票结果
            return "正面" if pos_count > neg_count else "负面"
        return None
