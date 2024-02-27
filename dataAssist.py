import pandas as pd
import numpy as np

class DataAssistMatrix:
    def __init__(self, params):
        self.params = params
        print('Loading data...')
        root = params.get('root', 'C:/Users/82108/OneDrive/바탕 화면/Pythonworkspace/ExploreCSR/DKT/DKTpython/data/assistments/')
        train_path = root + 'builder_train.csv'
        test_path = root + 'builder_test.csv'

        self.train_data = self.load_data(train_path)
        self.test_data = self.load_data(test_path)

        # Calculate the total number of unique questions
        all_questions = pd.concat([self.train_data['question_id'].explode(), self.test_data['question_id'].explode()])
        self.n_questions = all_questions.unique().shape[0]

        total_answers = self.train_data['n_answers'].sum() + self.test_data['n_answers'].sum()
        longest = max(self.train_data['n_answers'].max(), self.test_data['n_answers'].max())

        print('Total answers:', total_answers)
        print('Longest:', longest)
        print('Unique questions:', self.n_questions)


    def load_data(self, path):
        df = pd.DataFrame(columns=['n_answers', 'question_id', 'correct'])

        with open(path, 'r') as file:
            while True:
                lines = [next(file, None) for _ in range(3)]
                if None in lines:
                    break

                n_answers = int(lines[0].strip())
                # Filter out empty strings before converting to integers
                question_id = [int(i) + 1 for i in lines[1].strip().split(',') if i.strip()]
                correct = [int(i) for i in lines[2].strip().split(',') if i.strip()]

                df = pd.concat([df, pd.DataFrame({'n_answers': [n_answers], 'question_id': [question_id], 'correct': [correct]})], ignore_index=True)

        return df

    def get_test_data(self):
        return self.test_data

    def get_train_data(self):
        return self.train_data

    def get_test_batch(self):
        return self.test_data.to_dict('records')