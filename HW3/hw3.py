import spacy
from spacy.training import Example, offsets_to_biluo_tags
import random
import os

# 讀取並合併資料
def load_and_merge_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentence = []
            labels = []
            for line in file:
                line = line.strip()
                if not line:
                    if sentence:
                        data.append((sentence, labels))
                        sentence = []
                        labels = []
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        word, label = parts
                        sentence.append(word)
                        labels.append(label)
            if sentence:
                data.append((sentence, labels))
    return data

# 準備資料
def prepare_data(nlp, data):
    examples = []
    for sentence, labels in data:
        doc = nlp.make_doc(' '.join(sentence))
        entities = []
        start_idx = 0
        for i, label in enumerate(labels):
            if label != 'O':
                end_idx = start_idx + len(sentence[i])
                entities.append((start_idx, end_idx, label))
            start_idx += len(sentence[i]) + 1  # 加 1 因為有空格
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        examples.append(Example.from_dict(doc, {"entities": entities}))
    return examples

# 訓練 NER 模型
def train_ner_model(nlp, train_data, n_iter=10):
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for example in train_data:
        for ent in example.reference.ents:
            ner.add_label(ent.label_)

    optimizer = nlp.begin_training()

    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for example in train_data:
            nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
        # print(losses)

    return nlp

# 評估模型
def evaluate_ner_model(nlp, test_data):
    scorer = nlp.evaluate(test_data)
    return scorer

def print_evaluation_scores(scores):
    print("Token-level evaluation:")
    print(f"  Accuracy: {scores['token_acc']:.2f}")
    print(f"  Precision: {scores['token_p']:.2f}")
    print(f"  Recall: {scores['token_r']:.2f}")
    print(f"  F1-Score: {scores['token_f']:.2f}")
    
    print("\nEntity-level evaluation:")
    print(f"  Precision: {scores['ents_p']:.2f}")
    print(f"  Recall: {scores['ents_r']:.2f}")
    print(f"  F1-Score: {scores['ents_f']:.2f}")
    
    print("\nEntity-level evaluation per type:")
    for ent_type, ent_scores in scores['ents_per_type'].items():
        print(f"  {ent_type}:")
        print(f"    Precision: {ent_scores['p']:.2f}")
        print(f"    Recall: {ent_scores['r']:.2f}")
        print(f"    F1-Score: {ent_scores['f']:.2f}")

# 主程序
def main():
    # 指定所有文件路徑
    file_paths = [
        'HW3/data/a.conll',
        'HW3/data/b.conll',
        'HW3/data/e.conll',
        'HW3/data/f.conll',
        'HW3/data/g.conll',
        'HW3/data/h.conll'
    ]

    # 合併資料
    data = load_and_merge_data(file_paths)
    
    # 創建一個空白的NLP對象，用於準備資料
    nlp = spacy.blank('en')

    # 準備資料
    examples = prepare_data(nlp, data)
    train_examples = examples[:int(0.8 * len(examples))]
    test_examples = examples[int(0.8 * len(examples)):]

    # 訓練模型 
    nlp = train_ner_model(nlp, train_examples)

    # 保存模型
    model_path = "HW3/ner_model"
    nlp.to_disk(model_path)
    # print(f"Model saved to {model_path}")

    # 評估模型
    nlp = spacy.load(model_path)
    scores = evaluate_ner_model(nlp, test_examples)
    # print("Evaluation scores:", scores)
    print_evaluation_scores(scores)

if __name__ == "__main__":
    main()