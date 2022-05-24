# dataset name
dataset = "ml-1m"
assert dataset in ["ml-1m", "pinterest-20"], "데이터셋 이름 오류입니다."

# model name
model = "NeuMF-end"
assert model in ["MF", "MLP", "GMF", "NeuMF-end", "NeuMF-pre"], "모델 이름 오류입니다."

# paths
main_path = "data/"

train_rating = main_path + "{}.train.rating".format(dataset)
test_rating = main_path + "{}.test.rating".format(dataset)
test_negative = main_path +"{}.test.negative".format(dataset)

model_path = "models/"
MF_model_path = model_path + "MF.pth"
GMF_model_path = model_path + "GMF.pth"
MLP_model_path = model_path + "MLP.pth"
NeuMF_model_path = model_path + "NeuMF.pth"