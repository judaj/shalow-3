import pickle
import sys
from bilstmTrain import BilstmTagger
from bilstmTrain import DataParser

SUFF = 3
PREF = 3

def main():
    rper, model_path, test_file_name, train_data_file_name, pred_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    train_data = pickle.load(open(train_data_file_name, 'rb'))
    test_data = []
    sentence = []
    with open(test_file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                test_data.append(sentence)
                sentence = []
                continue
            sentence.append(line)

    lstm = BilstmTagger(train_data, rper)
    lstm.load(model_path)

    predictions = [lstm.predict_data(x, val=True) for x in test_data]
    with open(pred_path, 'w') as pred_file:
        for x, y in zip(test_data, predictions):
            for x1, y1 in zip(x, y):
                pred_file.write(x1 + ' ' + y1 + '\n')
            pred_file.write('\n')

if __name__ == "__main__":
    main()

