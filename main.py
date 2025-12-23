import os
import pandas as pd
from models import init_reviews
from models import tradition_models
from models import CNN
from models import BERT
from models import TextPredictor

# 定位文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
cn_negative_path = os.path.join(current_dir, 'evaltask2_sample_data/cn_sample_data/sample.negative.txt')
cn_positive_path = os.path.join(current_dir, 'evaltask2_sample_data/cn_sample_data/sample.positive.txt')
cn_test_path = os.path.join(current_dir, 'Sentiment Classification with Deep Learning/test.label.cn.txt')

en_negative_path = os.path.join(current_dir, 'evaltask2_sample_data/en_sample_data/sample.negative.txt')
en_positive_path = os.path.join(current_dir, 'evaltask2_sample_data/en_sample_data/sample.positive.txt')
en_test_path = os.path.join(current_dir, 'Sentiment Classification with Deep Learning/test.label.en.txt')


def train_model(language='Chinese'):
    if language == 'Chinese':
        print("正在训练中文模型...")
        reviews = init_reviews(cn_positive_path, cn_negative_path, cn_test_path, 'cn')
        X_train, y_train, X_test, y_test = reviews.get()
    elif language == 'English':
        print("正在训练英文模型...")
        reviews = init_reviews(en_positive_path, en_negative_path, en_test_path, 'en')
        X_train, y_train, X_test, y_test = reviews.get()
    elif language == 'Bilingual':
        print("正在训练双语模型...")
        cn_reviews = init_reviews(cn_positive_path, cn_negative_path, cn_test_path, 'cn')
        en_reviews = init_reviews(en_positive_path, en_negative_path, en_test_path, 'en')
        X_train = pd.concat([cn_reviews.train_df['review'],en_reviews.train_df['review']], ignore_index=True)
        y_train = pd.concat([cn_reviews.train_df['label'],en_reviews.train_df['label']], ignore_index=True)
        X_test = pd.concat([cn_reviews.test_df['review'],en_reviews.test_df['review']], ignore_index=True)
        y_test = pd.concat([cn_reviews.test_df['label'],en_reviews.test_df['label']], ignore_index=True)

    '''传统模型'''
    tradition_model = tradition_models(X_train, y_train, X_test, y_test, language)
    tradition_model.NaiveBayes()
    tradition_model.SVM()
    tradition_model.RandomForest()

    '''CNNM模型'''
    CNN_model = CNN(X_train, y_train, X_test, y_test, name=language)
    CNN_model.train()

    '''BERT模型'''
    bert_map = {
        'Chinese': 'bert-base-chinese', # 中文预训练库
        'English': 'bert-base-uncased', # 英文预训练库
        'Bilingual': 'bert-base-multilingual-cased' # 多语言预训练库
    }
    BERT_model = BERT(X_train, y_train, X_test, y_test, model_name=bert_map[language], name=language)
    BERT_model.train()

def main():
    check = {'1': '', '2': '', '3': ''}
    language_map = {
        '1': 'Chinese',
        '2': 'English',
        '3': 'Bilingual'
    }
    while True:
        """显示命令行菜单"""
        print("\n" + "="*50)
        print("文本情感分析模型训练系统")
        print("-"*50)
        print("请选择要训练的语言模型：")
        print(f"  1. 中文模型 (Chinese) {check['1']}")
        print(f"  2. 英文模型 (English) {check['2']}")
        print(f"  3. 双语模型 (Bilingual - 中英文混合) {check['3']}")
        print("  4. 训练以上模型")
        print("  5. 打开预测器")
        print("  6. 删除模型")
        print("  0. 退出程序")
        print("="*50)

        choice = get_user_choice(6)
        if choice == '0':
            print("程序退出")
            break
        elif choice in ['1', '2', '3']:
            if check[choice] == '':
                check[choice] = '√'
            else:
                check[choice] = ''
        elif choice == '4':
            list = [language_map[n] for n, c in check.items() if c == '√']
            if not list:
                print("\n请勾选需要训练的模型！")
                continue
            confirm = input(f"\n确认开始训练{list}吗（Y/N）：")
            if confirm.lower() == 'y':
                check = {'1': '', '2': '', '3': ''}
                for model_name in list:
                    train_model(model_name)
            else:
                continue
        elif choice == '5':
            predictor()
        elif choice == '6':
            model_del()

def get_user_choice(max_num):
    """获取用户选择"""
    while True:
        try:
            print("\n"+"/"*50)
            choice = input(f"\n请输入您的选择 (0-{max_num}): ").strip()
            if int(choice) in range(0, max_num+1):
                print("\n"+"/"*50)
                return choice
            else:
                print(f"无效输入，请重新输入 0-{max_num} 之间的数字")
        except ValueError:
            print("请输入有效数字！")
        print("\n"+"/"*50)

def predictor():
    """交互式预测器"""
    print("\n" + "="*50)
    print("文本情感分析预测器")
    print("-"*50)
    print("请选择语言模型:")
    print("1. 中文模型 (Chinese)")
    print("2. 英文模型 (English)")
    print("3. 双语模型 (Bilingual)")
    print("0. 退出预测器")
    print("="*50)
    
    language_choice = get_user_choice(3)

    if language_choice == '0':
        return
    language_map = {
        '1': 'Chinese',
        '2': 'English',
        '3': 'Bilingual'
    }
    
    language = language_map[language_choice]
    
    # 初始化预测器
    try:
        predictor = TextPredictor(language)
    except Exception as e:
        print(f"模型加载失败: {e}")
        choice = input(f"是否需要重新训练{language}模型(Y/N)：")
        if choice.lower() == 'y':
            train_model(language)
        return
    
    print(f"\n已加载{language}模型，请输入文本进行情感分析 (输入'quit'退出)")
    
    while True:
        print("\n" + "-"*50)
        text = input("\n请输入文本（输入quit退出）: ").strip()
        print("\n" + "-"*50)
        
        if text.lower() == 'quit':
            print("预测器退出")
            break
        
        if not text:
            print("输入不能为空")
            continue
        
        # 选择预测模式
        print("\n请选择预测模式:")
        print("1. 使用所有模型预测")
        print("2. 使用指定模型预测")
        print("0. 退出预测器")
        
        mode_choice = get_user_choice(2)
        
        if mode_choice == '1':
            # 使用所有模型预测
            final_result = predictor.predict_all(text)
            if final_result:
                print(f"\n最终综合预测结果: {final_result}")
            else:
                print("预测失败，请检查模型是否已训练")
        
        elif mode_choice == '2':
            # 使用指定模型预测
            print("\n请选择模型:")
            print("1. Naive Bayes (朴素贝叶斯)")
            print("2. SVM (支持向量机)")
            print("3. Random Forest (随机森林)")
            print("4. CNN (卷积神经网络)")
            print("5. BERT")
            print("0. 退出预测器")

            model_choice = get_user_choice(5)
            if mode_choice == '0':
                print("预测器退出")
                break
            
            model_map = {
                '1': ('Naive Bayes', predictor.predict_traditional),
                '2': ('SVM', predictor.predict_traditional),
                '3': ('Random Forest', predictor.predict_traditional),
                '4': ('CNN', lambda t: predictor.predict_cnn(t)),
                '5': ('BERT', lambda t: predictor.predict_bert(t))
            }
            
            model_name, predict_func = model_map[model_choice]
            if model_choice in ['1', '2', '3']:
                result = predict_func(text, model_name)
            else:
                result = predict_func(text)
            
            if result is not None:
                print(f"{model_name}预测结果: {result}")
            else:
                print(f"{model_name}模型未找到或加载失败")

def model_del():
    check = {'1': '', '2': '', '3': ''}
    language_map = {
        '1': 'Chinese',
        '2': 'English',
        '3': 'Bilingual'
    }
    while True:
        print("\n请选择要删除的模型:")
        print(f"1. 中文模型 {check['1']}")
        print(f"2. 英文模型 {check['2']}")
        print(f"3. 双语模型 {check['3']}")
        print("4. 确认")
        print("0. 退出")
        choice = get_user_choice(4)
        if choice in ['1', '2', '3']:
            if check[choice] == '':
                check[choice] = '√'
            else:
                check[choice] = ''
        elif choice == '4':
            list = [language_map[n] for n, c in check.items() if c == '√']
            if not list:
                print("\n请勾选需要删除的模型！")
                continue
            confirm = input(f"确认要删除{list}吗（Y/N）：")
            if confirm.lower() == 'y':
                import shutil
                for model_name in list:
                    try:
                        shutil.rmtree(os.path.join(current_dir, 'models', model_name))
                        print(f"\n{model_name}模型已删除")
                    except Exception as e:
                        print(f"\n{model_name}删除失败：{e}")
                    os.mkdir(os.path.join(current_dir, 'models', model_name))
                break
            else:
                continue
        elif choice == '0':
            break

if __name__ == '__main__':
    main()