import pickle
import os

if __name__ == "__main__":
    save_directory = os.getcwd()

    file_path = os.path.join(save_directory, '../evaluate_metrics/plain_225.pkl')
    # print(file_path)
    with open(file_path, 'rb') as file:
        res = pickle.load(file)
    
    print(res)