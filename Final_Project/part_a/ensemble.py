# TODO: complete this file.
from utils import *
from knn import *
from item_response import *
from neural_network import *

train_data = load_train_csv("../data")
sparse_matrix = load_train_sparse("../data").toarray()
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

# knn model
knn_prob = np.zeros(len(test_data["is_correct"]))


nbrs = KNNImputer(n_neighbors=11)
mat = nbrs.fit_transform(sparse_matrix)
for i in range(len(test_data["is_correct"])):
	cur_user_id = test_data["user_id"][i]
	cur_question_id = test_data["question_id"][i]
	knn_prob[i] = mat[cur_user_id, cur_question_id]

print(knn_prob)

# item response
item_response_prob = np.zeros(len(test_data["is_correct"]))
lr = 0.01
iter = 30
theta, beta, val_acc_lst,  train_neg_lld_list, val_neg_lld_list = irt(train_data, val_data, lr, iter)
for i, q in enumerate(test_data["question_id"]):
	u = test_data["user_id"][i]
	x = (theta[u] - beta[q]).sum()
	p_a = sigmoid(x)
	item_response_prob[i] = p_a
print(item_response_prob)

# neural_network
neural_network_prob = np.zeros(len(test_data["is_correct"]))
k = 10
zero_train_matrix, train_matrix, valid_data, test_data = load_data()
model = AutoEncoder(num_question=len(zero_train_matrix[0]), k=k)

lr = 0.01
num_epoch = 100
lamb = 1

loss_list, val_acc = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
model.eval()

for i, u in enumerate(test_data["user_id"]):
	inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
	output = model(inputs)
	neural_network_prob[i] = output[0][test_data["question_id"][i]].item()
print(neural_network_prob)

# ensemble model
prob = (knn_prob + item_response_prob + neural_network_prob) / 3
correct = 0
for i in range(len(prob)):
	if (prob[i] >= 0.5) == test_data["is_correct"][i]:
		correct += 1


print("final test accuracy:", correct/len(prob)*1.0)



