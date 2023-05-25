import fasttext

if __name__ == '__main__':
    model = fasttext.train_unsupervised(input='data/fasttext_train.txt', dim=300, thread=30)
    model.save_model("data/fasttext_custom.bin")